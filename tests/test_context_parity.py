"""Hand-written MMR and edge ordering against LangChain's. Skipped when the `frameworks`
extra is not installed."""

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("langchain_core")
pytest.importorskip("langchain_community")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "framework_equivalents"))

from context_assembly import langchain_mmr, langchain_reorder  # noqa: E402

from ragbasics.context.assemble import order_edges  # noqa: E402
from ragbasics.context.mmr import mmr  # noqa: E402
from ragbasics.index.numpy_store import unit_rows  # noqa: E402
from ragbasics.types import Chunk, Passage  # noqa: E402


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("lam", [0.0, 0.3, 0.5, 0.7, 1.0])
def test_mmr_picks_what_langchain_picks_given_the_query_cosine(seed, lam):
    rng = np.random.default_rng(seed)
    # Clustered vectors, so redundancy matters: 30 candidates around 6 centres.
    centres = rng.normal(size=(6, 32))
    vectors = unit_rows(centres[rng.integers(0, 6, 30)] + 0.3 * rng.normal(size=(30, 32)))
    query = unit_rows(rng.normal(size=(1, 32)))[0]
    mine = mmr(vectors @ query, vectors @ vectors.T, 8, lam)
    assert mine == langchain_mmr(query, vectors, 8, lam)


@pytest.mark.parametrize("n", range(1, 9))
def test_edge_order_is_langchains_long_context_reorder(n):
    passages = [Passage((Chunk(f"c{i}", "d", "x", i, i + 1),), rank=i + 1) for i in range(n)]
    assert order_edges(passages) == langchain_reorder(passages)
