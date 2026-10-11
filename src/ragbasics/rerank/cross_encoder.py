"""Cross-encoder reranking: one forward pass per (question, passage) pair.

An embedding model reads the question and the passage separately and compares two
vectors. A cross-encoder reads them as one input, so every word of the question can
attend to every word of the passage. That is more accurate, and nothing about a passage
can be computed before the question arrives, which is why it only runs on a short list.

The weights are a download and the forward pass is the transformers library's. What is
written here, where it can be read:

- pairing each candidate with the question, and what text the model is shown;
- truncation: the model reads at most `max_length` tokens and cuts longer pairs without
  any error, so pairs over the limit are counted in `truncated_pairs`;
- batching: pairs are sorted by length so that a batch is padded to its own longest
  pair and not to the longest pair of all;
- turning the model's output into one score per pair, for two kinds of model;
- sorting, with ties left in first-stage order;
- a disk cache of scores.

Two kinds of model:

    classifier  an encoder with a one-number head, trained on (query, passage, label)
                triples. The score is that number (a logit; higher is more relevant).
                ms-marco-MiniLM and bge-reranker-v2-m3 are this kind.
    yes_no      a causal language model given a prompt that asks whether the passage
                answers the query. The score is the logit of "yes" minus the logit of
                "no" for the next token. Qwen3-Reranker is this kind.

PyTorch and transformers are in the "local" extra and are imported on first use.
"""

import threading
import time
from pathlib import Path
from typing import Any, Protocol

from ragbasics.registry import register
from ragbasics.rerank.base import Reranked, passage_text, reorder
from ragbasics.rerank.cache import RerankCache, pair_key
from ragbasics.types import Candidate

Pair = tuple[str, str]  # (question, passage)

# The prompt Qwen3-Reranker was trained with, from its model card.
YES_NO_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the "
    'Query and the Instruct provided. Note that the answer can only be "yes" or "no".'
    "<|im_end|>\n<|im_start|>user\n"
)
YES_NO_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
YES_NO_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"


def pick_device(device: str | None) -> str:
    if device:
        return device
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


class PairScorer(Protocol):
    """The model: token counts and one forward pass. Everything else is the reranker's."""

    identity: str  # everything besides the pair that decides a score
    max_length: int

    def lengths(self, pairs: list[Pair]) -> list[int]: ...

    def forward(self, pairs: list[Pair]) -> list[float]: ...


class ClassifierScorer:
    def __init__(self, model: str, revision: str | None, max_length: int, device: str | None):
        self.model, self.revision, self.max_length = model, revision, max_length
        self.device = device
        self.identity = f"classifier:{model}@{revision}:{max_length}"
        self._loaded: tuple[Any, Any] | None = None
        self.load_seconds = 0.0

    def load(self) -> tuple[Any, Any]:
        if self._loaded is None:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            started = time.perf_counter()
            self.device = pick_device(self.device)
            tokenizer = AutoTokenizer.from_pretrained(self.model, revision=self.revision)
            network = AutoModelForSequenceClassification.from_pretrained(
                self.model, revision=self.revision
            )
            network.to(self.device).eval()
            self._loaded = (tokenizer, network)
            self.load_seconds = time.perf_counter() - started
        return self._loaded

    def lengths(self, pairs: list[Pair]) -> list[int]:
        tokenizer, _ = self.load()
        ids = tokenizer([q for q, _ in pairs], [p for _, p in pairs], truncation=False)
        return [len(row) for row in ids["input_ids"]]

    def forward(self, pairs: list[Pair]) -> list[float]:
        import torch

        tokenizer, network = self.load()
        # One input per pair: [CLS] question [SEP] passage [SEP]. When a pair is over the
        # limit, tokens are dropped from the longer of the two texts.
        batch = tokenizer(
            [q for q, _ in pairs],
            [p for _, p in pairs],
            padding=True,
            truncation="longest_first",
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            logits = network(**batch).logits
        return logits[:, 0].float().cpu().tolist()


class YesNoScorer:
    def __init__(
        self,
        model: str,
        revision: str | None,
        max_length: int,
        device: str | None,
        instruction: str = YES_NO_INSTRUCTION,
    ):
        self.model, self.revision, self.max_length = model, revision, max_length
        self.device = device
        self.instruction = instruction
        self.identity = (
            f"yes_no:{model}@{revision}:{max_length}:{YES_NO_PREFIX}:{instruction}:{YES_NO_SUFFIX}"
        )
        self._loaded: tuple[Any, Any] | None = None
        self.load_seconds = 0.0

    def load(self) -> tuple[Any, Any]:
        if self._loaded is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            started = time.perf_counter()
            self.device = pick_device(self.device)
            # Padding goes on the left: the score is read at the last position, which
            # must be the end of the prompt for every row of the batch.
            tokenizer = AutoTokenizer.from_pretrained(
                self.model, revision=self.revision, padding_side="left"
            )
            # float32 on purpose. This model's config asks for 16-bit weights, which
            # rounds the two logits to steps of about 0.1 and leaves many passages tied.
            network = AutoModelForCausalLM.from_pretrained(
                self.model, revision=self.revision, dtype="float32"
            )
            network.to(self.device).eval()
            self._loaded = (tokenizer, network)
            self._prefix = tokenizer.encode(YES_NO_PREFIX, add_special_tokens=False)
            self._suffix = tokenizer.encode(YES_NO_SUFFIX, add_special_tokens=False)
            self._yes = tokenizer.convert_tokens_to_ids("yes")
            self._no = tokenizer.convert_tokens_to_ids("no")
            self.load_seconds = time.perf_counter() - started
        return self._loaded

    def body(self, pair: Pair) -> str:
        question, passage = pair
        return f"<Instruct>: {self.instruction}\n<Query>: {question}\n<Document>: {passage}"

    def lengths(self, pairs: list[Pair]) -> list[int]:
        tokenizer, _ = self.load()
        bodies = tokenizer([self.body(p) for p in pairs], add_special_tokens=False)["input_ids"]
        return [len(self._prefix) + len(ids) + len(self._suffix) for ids in bodies]

    def forward(self, pairs: list[Pair]) -> list[float]:
        import torch

        tokenizer, network = self.load()
        # The fixed prompt around the pair is never cut; the pair itself is cut to fit.
        room = self.max_length - len(self._prefix) - len(self._suffix)
        bodies = tokenizer(
            [self.body(p) for p in pairs], add_special_tokens=False, truncation=True,
            max_length=room,
        )["input_ids"]
        rows = [self._prefix + ids + self._suffix for ids in bodies]
        batch = tokenizer.pad({"input_ids": rows}, padding=True, return_tensors="pt")
        batch = batch.to(self.device)
        with torch.inference_mode():
            # Only the last position is needed, so only it goes through the output
            # layer. Asking the model for `.logits` would compute a score for every
            # word of the vocabulary at every position of every pair.
            hidden = network.base_model(**batch).last_hidden_state[:, -1, :]
            last = network.get_output_embeddings()(hidden)
        return (last[:, self._yes] - last[:, self._no]).float().cpu().tolist()


SCORERS = {"classifier": ClassifierScorer, "yes_no": YesNoScorer}


@register("reranker", "cross_encoder")
class CrossEncoderReranker:
    """
    model, revision  the Hugging Face model and the pinned commit.
    scoring          classifier or yes_no (see the module docstring).
    depth            how many first-stage candidates are read and reordered.
    max_length       the most tokens the model reads per pair.
    header           also show the model each chunk's title, source and date.
    cache            a SQLite file of scores already computed.
    """

    def __init__(
        self,
        model: str,
        revision: str | None = None,
        scoring: str = "classifier",
        depth: int = 50,
        max_length: int = 512,
        batch_size: int = 32,
        header: bool = False,
        instruction: str | None = None,
        device: str | None = None,
        cache: str | Path | None = None,
        scorer: PairScorer | None = None,
    ):
        if scoring not in SCORERS:
            raise ValueError(f"unknown scoring '{scoring}'; choose from {sorted(SCORERS)}")
        if depth < 1:
            raise ValueError(f"depth must be at least 1; got {depth}")
        self.model = model
        self.depth = depth
        self.batch_size = batch_size
        self.header = header
        if scorer is None:
            extra = {"instruction": instruction} if scoring == "yes_no" and instruction else {}
            scorer = SCORERS[scoring](model, revision, max_length, device, **extra)
        self.scorer = scorer
        self._cache_path = Path(cache) if cache else None
        self._cache: RerankCache | None = None
        self.truncated_pairs = 0  # pairs longer than the model reads
        # The evaluator reranks from several threads; one forward pass at a time.
        self._lock = threading.Lock()

    @property
    def cache(self) -> RerankCache | None:
        # Opened on first use, so building a pipeline creates no file.
        if self._cache is None and self._cache_path:
            self._cache = RerankCache(self._cache_path)
        return self._cache

    def scores(self, pairs: list[Pair]) -> tuple[list[float], int]:
        """One score per pair, and how many of them the model had to compute now."""
        keys = [pair_key(self.scorer.identity, q, p) for q, p in pairs]
        cache = self.cache  # an empty cache has length 0, so it is compared with None
        known = cache.get_many(keys) if cache is not None else {}
        # A pair that appears twice is scored once.
        missing: dict[str, Pair] = {}
        for key, pair in zip(keys, pairs, strict=True):
            if key not in known:
                missing.setdefault(key, pair)
        if missing:
            with self._lock:
                fresh = dict(zip(missing, self._compute(list(missing.values())), strict=True))
            if cache is not None:
                cache.put_many(fresh)
            known |= fresh
        return [known[key] for key in keys], len(missing)

    def _compute(self, pairs: list[Pair]) -> list[float]:
        lengths = self.scorer.lengths(pairs)
        self.truncated_pairs += sum(n > self.scorer.max_length for n in lengths)
        # Shortest first, so each batch is padded only to its own longest pair.
        order = sorted(range(len(pairs)), key=lambda i: lengths[i])
        scores = [0.0] * len(pairs)
        for start in range(0, len(order), self.batch_size):
            batch = order[start : start + self.batch_size]
            for i, score in zip(batch, self.scorer.forward([pairs[i] for i in batch]), strict=True):
                scores[i] = score
        return scores

    def rerank(self, question: str, candidates: list[Candidate]) -> Reranked:
        started = time.perf_counter()
        truncated_before = self.truncated_pairs
        pairs = [(question, passage_text(c.chunk, self.header)) for c in candidates]
        scores, computed = self.scores(pairs) if pairs else ([], 0)
        return Reranked(
            candidates=reorder(candidates, scores, "rerank"),
            timings={"rerank": time.perf_counter() - started},
            cached=computed == 0,
            notes={"truncated_pairs": self.truncated_pairs - truncated_before},
        )
