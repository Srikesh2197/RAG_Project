"""The hand-written cross-encoder reranker against sentence-transformers' CrossEncoder.

Skipped when the `local` extra is not installed, and for any model whose weights are not
already on this machine: a test never downloads a model.
"""

import sys
from pathlib import Path

import pytest

pytest.importorskip("sentence_transformers")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "framework_equivalents"))

import cross_encoder_library  # noqa: E402
import yaml  # noqa: E402
from huggingface_hub import try_to_load_from_cache  # noqa: E402

from ragbasics.rerank.cross_encoder import CrossEncoderReranker  # noqa: E402
from test_bm25_parity import SENTENCES  # noqa: E402

CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "stage06_rerank"
QUESTIONS = [
    "Why was the harbour ferry delayed?",
    "What closed the airport, and for how many days?",
    "Which bakery chain shares rose after a strong quarter?",
]
# Passages of one to four sentences, so batches hold pairs of different lengths.
PASSAGES = [" ".join(SENTENCES[i % 8 : i % 8 + 1 + i % 4]) for i in range(24)]


def settings(stem: str) -> dict:
    return yaml.safe_load((CONFIGS / f"{stem}.yaml").read_text())["reranker"]["params"]


def on_disk(params: dict) -> bool:
    found = try_to_load_from_cache(params["model"], "config.json", revision=params["revision"])
    return isinstance(found, str)


@pytest.mark.parametrize("stem", ["minilm-50", "bge-v2-m3-50"])
def test_classifier_scores_and_order_match_sentence_transformers(stem):
    params = settings(stem)
    if not on_disk(params):
        pytest.skip(f"{params['model']} is not downloaded")
    mine = CrossEncoderReranker(
        params["model"], revision=params["revision"], batch_size=7, max_length=64
    )
    library = cross_encoder_library.load(params["model"], params["revision"], max_length=64)
    for question in QUESTIONS:
        pairs = [(question, passage) for passage in PASSAGES]
        ours, _ = mine.scores(pairs)
        theirs = cross_encoder_library.scores(library, pairs)
        # Batches are padded differently on the two sides, which moves the last digits.
        assert ours == pytest.approx(theirs, abs=2e-3)
        order = sorted(range(len(ours)), key=lambda i: (-round(ours[i], 2), i))
        ranked = cross_encoder_library.rank(library, question, PASSAGES)
        assert order == sorted(ranked, key=lambda i: (-round(theirs[i], 2), i))
    # max_length 64 is below the longest pair here, so truncation is exercised and
    # counted, and both sides cut the same tokens.
    assert mine.truncated_pairs > 0


def test_yes_no_scores_match_the_recipe_on_the_qwen3_model_card():
    """Qwen3-Reranker's card gives its scoring as code: build the prompt, read the
    logits of "no" and "yes" at the last position, and take a softmax over the two.
    The hand-written scorer returns yes minus no, whose sigmoid is that probability."""
    import math

    import torch

    params = settings("qwen3-50")
    if not on_disk(params):
        pytest.skip(f"{params['model']} is not downloaded")
    mine = CrossEncoderReranker(
        params["model"], revision=params["revision"], scoring="yes_no", batch_size=4
    )
    pairs = [(QUESTIONS[0], passage) for passage in PASSAGES[:8]]
    ours, _ = mine.scores(pairs)

    scorer = mine.scorer
    tokenizer, network = scorer.load()
    prefix = tokenizer.encode(
        "<|im_start|>system\nJudge whether the Document meets the requirements based on "
        'the Query and the Instruct provided. Note that the answer can only be "yes" or '
        '"no".<|im_end|>\n<|im_start|>user\n', add_special_tokens=False,
    )
    suffix = tokenizer.encode(
        "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", add_special_tokens=False
    )
    instruction = "Given a web search query, retrieve relevant passages that answer the query"
    texts = [f"<Instruct>: {instruction}\n<Query>: {q}\n<Document>: {p}" for q, p in pairs]
    inputs = tokenizer(texts, padding=False, return_attention_mask=False)
    inputs["input_ids"] = [prefix + ids + suffix for ids in inputs["input_ids"]]
    inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt").to(scorer.device)
    with torch.no_grad():
        last = network(**inputs).logits[:, -1, :]
    no, yes = (tokenizer.convert_tokens_to_ids(word) for word in ("no", "yes"))
    theirs = torch.softmax(torch.stack([last[:, no], last[:, yes]], dim=1).float(), dim=1)[:, 1]
    assert [1 / (1 + math.exp(-score)) for score in ours] == pytest.approx(
        theirs.tolist(), abs=1e-3
    )
