"""Listwise reranking with an LLM: one prompt holds the question and every candidate,
and the model replies with an ordering.

A cross-encoder scores each passage on its own. A listwise reranker sees the passages
next to each other, so it can prefer the one that states a fact over five that mention
it. The price is a paid call per question whose prompt grows with the number of
candidates, and a reply that is text and has to be parsed.

Written here: the prompt, the parsing, and the repair of a reply that is not a clean
ordering (a number repeated, left out, or out of range). Passages the model left out
keep their first-stage order below the ones it ranked.

A provider can refuse a prompt outright: news passages about crime or violence can trip a
usage-policy filter. A refused question keeps its first-stage order and is counted in
the trace as `refused`; nothing is cached for it. Any other error is raised.

There is no score per passage, only a position. `Candidate.score` is 1 / new rank, so
that a higher score still means a better passage.
"""

import re
import time
from pathlib import Path
from typing import Any

import ragbasics.generation.llm  # noqa: F401  (registers the generators)
from ragbasics.cache import CachedGenerator, DiskCache
from ragbasics.generation.llm import Generator
from ragbasics.registry import build, register
from ragbasics.rerank.base import Reranked, passage_text, reorder
from ragbasics.types import Candidate

SYSTEM_PROMPT = """\
You rank passages from news articles by how useful each is as evidence for answering a \
question. The user message contains numbered passages, followed by the question.

A passage is useful if it states a fact the answer depends on. A question may need \
facts from several passages; rank each of those highly. A passage that is only on the \
same topic is less useful than one that states a needed fact.

Reply with every passage number exactly once, most useful first, separated by " > ". \
For example: 3 > 1 > 2
Give the ranking alone, with no other text."""


def build_prompt(question: str, passages: list[str]) -> str:
    numbered = "\n\n".join(f"[{i}]\n{text.strip()}" for i, text in enumerate(passages, start=1))
    return f"{numbered}\n\nQuestion: {question}"


def parse_ranking(reply: str, n: int) -> tuple[list[int], dict[str, int]]:
    """The reply as a full ordering of passages 0 to n - 1, and what had to be repaired.

    Numbers are read in the order they appear. A number outside 1 to n is dropped, a
    repeat counts the first time only, and passages never mentioned are added at the
    end in their original order.
    """
    order: list[int] = []
    seen: set[int] = set()
    invalid = repeated = 0
    for match in re.findall(r"\d+", reply):
        number = int(match)
        if not 1 <= number <= n:
            invalid += 1
        elif number in seen:
            repeated += 1
        else:
            seen.add(number)
            order.append(number - 1)
    missing = [i for i in range(n) if i + 1 not in seen]
    return order + missing, {"invalid": invalid, "repeated": repeated, "missing": len(missing)}


@register("reranker", "llm_listwise")
class ListwiseReranker:
    """
    generator  the LLM, as in a pipeline config: {name: openai, params: {model: ...}}.
    depth      how many first-stage candidates go in the prompt.
    header     put each chunk's title, source and date above its text.
    cache      the SQLite file of LLM calls; a repeated prompt is not paid for again.
    """

    def __init__(
        self,
        generator: dict[str, Any] | Generator,
        depth: int = 20,
        header: bool = False,
        cache: str | Path | None = None,
    ):
        if depth < 1:
            raise ValueError(f"depth must be at least 1; got {depth}")
        if isinstance(generator, dict):
            generator = build("generator", generator["name"], **generator.get("params", {}))
        self.inner = generator
        self.model = generator.model
        self.depth = depth
        self.header = header
        self._cache_path = Path(cache) if cache else None
        self._llm: Generator | None = None

    @property
    def llm(self) -> Generator:
        # The cache file is opened on first use, so building a pipeline creates none.
        if self._llm is None:
            self._llm = (
                CachedGenerator(self.inner, DiskCache(self._cache_path))
                if self._cache_path
                else self.inner
            )
        return self._llm

    def prompt(self, question: str, candidates: list[Candidate]) -> str:
        return build_prompt(question, [passage_text(c.chunk, self.header) for c in candidates])

    def is_cached(self, question: str, candidates: list[Candidate]) -> bool:
        check = getattr(self.llm, "is_cached", None)
        return bool(check and check(SYSTEM_PROMPT, self.prompt(question, candidates)))

    def rerank(self, question: str, candidates: list[Candidate]) -> Reranked:
        if not candidates:
            return Reranked(candidates=[])
        started = time.perf_counter()
        try:
            generation = self.llm.generate(SYSTEM_PROMPT, self.prompt(question, candidates))
        except Exception as error:
            # The OpenAI and Anthropic SDKs both raise BadRequestError when the request
            # itself is rejected. Named by its class so neither SDK is imported here.
            if type(error).__name__ != "BadRequestError":
                raise
            return Reranked(
                candidates=reorder(candidates, [0.0] * len(candidates), "rerank"),
                timings={"rerank": time.perf_counter() - started},
                notes={"refused": True},
            )
        order, repairs = parse_ranking(generation.text, len(candidates))
        scores = [0.0] * len(candidates)
        for position, i in enumerate(order, start=1):
            scores[i] = 1.0 / position
        return Reranked(
            candidates=reorder(candidates, scores, "rerank"),
            timings={"rerank": time.perf_counter() - started},
            input_tokens=generation.input_tokens,
            output_tokens=generation.output_tokens,
            cached=generation.cached,
            notes=repairs | {"stop_reason": generation.stop_reason},
        )
