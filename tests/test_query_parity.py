"""The hand-written multi-query and HyDE transformations against LangChain's.

An LLM's output is not deterministic, so both sides are given the same scripted reply and
the test compares what is: the queries read out of the reply, and the chunks retrieved
with them. Skipped when the `frameworks` extra is not installed.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("langchain_classic")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "framework_equivalents"))

import query_transforms  # noqa: E402

from ragbasics.generation.llm import Generation  # noqa: E402
from ragbasics.pipeline import Pipeline  # noqa: E402
from ragbasics.query.llm import HyDE, MultiQuery  # noqa: E402
from ragbasics.types import Document  # noqa: E402
from test_bm25_parity import SENTENCES  # noqa: E402
from test_pipeline import offline_config  # noqa: E402

QUESTION = "Why was the harbour ferry delayed and what closed the airport?"
CLEAN = "What held up the harbour ferry?\nWhich event shut the airport?\nWhy did flights stop?"


class Scripted:
    model = "scripted"

    def __init__(self, reply: str):
        self.reply = reply

    def generate(self, system: str, user: str) -> Generation:
        return Generation(self.reply, stop_reason="stop")


@pytest.fixture
def pipeline(tmp_path):
    # One sentence per document, so there are enough chunks for the lists to differ.
    pipeline = Pipeline(offline_config(tmp_path, top_k=4))
    pipeline.ingest([Document(f"doc_{i:02d}", text) for i, text in enumerate(SENTENCES)])
    return pipeline


def test_multi_query_reads_the_same_queries_and_retrieves_the_same_chunks(pipeline):
    k = 4
    ours = MultiQuery(Scripted(CLEAN), n=3)
    pipeline.transformer = ours
    trace = pipeline.retrieve(QUESTION, k=k)
    library = query_transforms.multi_query(pipeline, CLEAN, k, ours.system_prompt())

    # The same queries from the same reply. The library searches the question last.
    theirs = library.llm_chain.invoke({"question": QUESTION})
    assert theirs == [q.text for q in trace.queries[1:]] == CLEAN.split("\n")

    # The library returns the union of every query's top k, each chunk once, in the
    # order first seen. The same chunks are in the hand-written per-query lists.
    documents = library.invoke(QUESTION)
    union = [d.id for d in documents]
    per_query = [[c.chunk.chunk_id for c in found[:k]] for found in trace.query_lists]
    first_seen = list(dict.fromkeys(
        chunk_id for found in [*per_query[1:], per_query[0]] for chunk_id in found
    ))
    assert union == first_seen and len(union) > k

    # What differs: the hand-written list is ranked across queries and cut to k. Its
    # chunks are among the union's, and a chunk several queries found comes first.
    fused = [c.chunk.chunk_id for c in trace.candidates]
    assert len(fused) == k and set(fused) <= set(union)
    counts = {chunk_id: sum(chunk_id in found for found in per_query) for chunk_id in union}
    assert counts[fused[0]] == max(counts.values())


def test_the_library_keeps_numbering_and_repeats_that_the_hand_written_parser_removes(pipeline):
    reply = "1. What held up the ferry?\n2. What held up the ferry?\n\n- Why did flights stop?"
    ours = MultiQuery(Scripted(reply), n=3).transform(QUESTION)
    assert [q.text for q in ours.queries] == ["What held up the ferry?", "Why did flights stop?"]
    assert ours.notes == {"repeated": 1, "missing": 1, "stop_reason": "stop"}
    library = query_transforms.multi_query(pipeline, reply, 4, "Rephrase.")
    assert library.llm_chain.invoke({"question": QUESTION}) == [
        "1. What held up the ferry?", "2. What held up the ferry?", "- Why did flights stop?",
    ]


def test_hyde_searches_with_the_same_vector_and_finds_the_same_chunks(pipeline):
    passage = "Thick fog held up the harbour ferry on Tuesday morning, the operator said."
    ours = HyDE(Scripted(passage))
    pipeline.transformer = ours
    trace = pipeline.retrieve(QUESTION, k=4)
    assert [q.text for q in trace.queries] == [passage]

    vector = query_transforms.hyde_vector(pipeline, passage, QUESTION, ours.system_prompt())
    expected = pipeline.embedder.embed([passage], kind="document").vectors[0]
    assert np.allclose(vector, expected)
    found = pipeline.store.search(np.asarray(vector, dtype=np.float32), 4)
    assert [c.chunk.chunk_id for c in found] == [c.chunk.chunk_id for c in trace.candidates]
