import numpy as np
import pytest

from ragbasics.index.numpy_store import NumpyStore, unit_rows
from ragbasics.types import Chunk


def make_chunks(n: int) -> list[Chunk]:
    return [Chunk(f"doc_0:{i:04d}", "doc_0", f"text {i}", i * 10, i * 10 + 6) for i in range(n)]


def test_unit_rows_scales_to_length_one_and_keeps_zero_rows():
    rows = unit_rows(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert rows[0] == pytest.approx([0.6, 0.8])
    assert rows[1] == pytest.approx([0.0, 0.0])


def test_search_matches_hand_computed_cosines():
    # Stored, after scaling:  a = (0.6, 0.8)   b = (1, 0)   c = (0, 1)
    # Query (1, 1) scales to (0.7071, 0.7071).
    #   cos(q, a) = 0.6 * 0.7071 + 0.8 * 0.7071 = 0.9899
    #   cos(q, b) = cos(q, c) = 0.7071, a tie, so insertion order decides: b then c.
    store = NumpyStore()
    store.add(make_chunks(3), np.array([[3.0, 4.0], [1.0, 0.0], [0.0, 2.0]]))
    results = store.search(np.array([1.0, 1.0]), k=3)
    assert [r.chunk.chunk_id for r in results] == ["doc_0:0000", "doc_0:0001", "doc_0:0002"]
    assert [r.score for r in results] == pytest.approx([0.98995, 0.70711, 0.70711], abs=1e-5)
    assert [r.rank for r in results] == [1, 2, 3]


def test_a_long_vector_wins_on_dot_product_and_loses_on_cosine():
    long_off_target = np.array([10.0, 0.0])
    short_on_target = np.array([0.6, 0.8])
    query = np.array([0.6, 0.8])
    # Raw dot product rewards length: 6.0 against 1.0.
    assert long_off_target @ query > short_on_target @ query

    store = NumpyStore()
    store.add(make_chunks(2), np.stack([long_off_target, short_on_target]))
    best = store.search(query, k=1)[0]
    assert best.chunk.chunk_id == "doc_0:0001"
    assert best.score == pytest.approx(1.0)


def test_k_larger_than_the_index_and_empty_index():
    store = NumpyStore()
    assert store.search(np.array([1.0, 0.0]), k=5) == []
    store.add(make_chunks(2), np.eye(2))
    assert len(store.search(np.array([1.0, 0.0]), k=5)) == 2


def test_add_accumulates_across_calls():
    store = NumpyStore()
    chunks = make_chunks(3)
    store.add(chunks[:2], np.array([[1.0, 0.0], [0.0, 1.0]]))
    store.add(chunks[2:], np.array([[-1.0, 0.0]]))
    assert len(store) == 3
    assert store.search(np.array([-1.0, 0.0]), k=1)[0].chunk.chunk_id == "doc_0:0002"


def test_mismatched_dimensions_are_rejected():
    store = NumpyStore()
    store.add(make_chunks(1), np.array([[1.0, 0.0]]))
    with pytest.raises(ValueError, match="dimensions"):
        store.search(np.array([1.0, 0.0, 0.0]), k=1)
    with pytest.raises(ValueError, match="dimensional"):
        store.add(make_chunks(1), np.array([[1.0, 0.0, 0.0]]))
    with pytest.raises(ValueError, match="chunks but"):
        store.add(make_chunks(2), np.array([[1.0, 0.0]]))


def test_save_and_load_round_trip(tmp_path):
    store = NumpyStore()
    store.add(make_chunks(3), np.array([[3.0, 4.0], [1.0, 0.0], [0.0, 2.0]]))
    store.save(tmp_path, {"embedder": "test"})

    loaded, meta = NumpyStore.load(tmp_path)
    assert meta == {"embedder": "test"}
    assert loaded.chunks == store.chunks
    query = np.array([1.0, 1.0])
    assert loaded.search(query, k=3) == store.search(query, k=3)
