"""Stage 4: truncated dimensions, the embedding cache, and query and document prefixes.

Nothing here loads a model: the local embedder is given a stand-in encoder, so the tests
run without sentence-transformers installed.
"""

import numpy as np
import pytest

from ragbasics.embedding.base import Embedded, truncate
from ragbasics.embedding.cache import EmbeddingCache
from ragbasics.embedding.local import LocalEmbedder
from ragbasics.embedding.openai_embedder import OpenAIEmbedder
from ragbasics.pipeline import Pipeline
from ragbasics.registry import register
from ragbasics.types import Document
from test_pipeline import StubOpenAI, offline_config

# --- Truncation ----------------------------------------------------------------------------


def test_truncate_keeps_the_leading_dimensions_and_rescales_to_length_one():
    # (3, 4, 12) has length 13. Its first two numbers, (3, 4), have length 5.
    vectors = np.array([[3.0, 4.0, 12.0]], dtype=np.float32) / 13.0
    cut = truncate(vectors, 2)
    assert cut[0].tolist() == pytest.approx([0.6, 0.8])
    assert np.linalg.norm(cut, axis=1) == pytest.approx([1.0])


def test_without_rescaling_a_truncated_dot_product_is_not_a_cosine():
    # Both vectors point the same way in their first two dimensions, so their truncated
    # cosine is 1. The raw truncated dot product is far below it, and by a different
    # amount for each vector, because each kept a different share of its length.
    a = np.array([[3.0, 4.0, 12.0]], dtype=np.float32) / 13.0
    b = np.array([[3.0, 4.0, 0.0]], dtype=np.float32) / 5.0
    assert float(a[0, :2] @ b[0, :2]) == pytest.approx(5 / 13)
    assert float(truncate(a, 2)[0] @ truncate(b, 2)[0]) == pytest.approx(1.0)


def test_truncate_passes_through_and_refuses_impossible_lengths():
    vectors = np.ones((2, 4), dtype=np.float32)
    assert truncate(vectors, None) is vectors
    for dimensions in (0, 5):
        with pytest.raises(ValueError, match="truncate"):
            truncate(vectors, dimensions)


def test_a_zero_vector_stays_zero_when_truncated():
    assert truncate(np.zeros((1, 4), dtype=np.float32), 2).tolist() == [[0.0, 0.0]]


# --- Cache ---------------------------------------------------------------------------------


def test_cache_round_trips_vectors_and_survives_reopening(tmp_path):
    path = tmp_path / "cache" / "embeddings.sqlite"
    cache = EmbeddingCache(path)
    cache.put_many({"a": np.array([1.0, 2.5], dtype=np.float32)})
    assert cache.get_many(["a", "missing"]).keys() == {"a"}
    reopened = EmbeddingCache(path)
    assert reopened.get_many(["a"])["a"].tolist() == [1.0, 2.5]
    assert len(reopened) == 1


def test_cache_looks_up_more_keys_than_one_sql_statement_holds(tmp_path):
    cache = EmbeddingCache(tmp_path / "e.sqlite")
    vectors = {f"k{i}": np.array([float(i)], dtype=np.float32) for i in range(1200)}
    cache.put_many(vectors)
    found = cache.get_many(list(vectors))
    assert len(found) == 1200 and found["k1199"].tolist() == [1199.0]


def test_cached_texts_are_not_embedded_or_billed_again(tmp_path):
    client = StubOpenAI()
    embedder = OpenAIEmbedder(client=client)
    embedder.cache = EmbeddingCache(tmp_path / "e.sqlite")
    first = embedder.embed(["a", "b"])
    assert first.tokens == 6 and client.calls == [["a", "b"]]

    second = embedder.embed(["b", "c", "a"])
    assert client.calls == [["a", "b"], ["c"]]  # only the new text went to the model
    assert second.tokens == 3
    assert second.vectors[2].tolist() == first.vectors[0].tolist()
    assert second.vectors[0].tolist() == first.vectors[1].tolist()

    # A new embedder on the same file pays nothing and gets the same numbers.
    fresh = OpenAIEmbedder(client=StubOpenAI())
    fresh.cache = EmbeddingCache(tmp_path / "e.sqlite")
    again = fresh.embed(["a", "b"])
    assert again.tokens == 0 and fresh.client.calls == []
    assert again.vectors.tolist() == first.vectors.tolist()


def test_uncached_lists_what_would_be_sent_to_the_model(tmp_path):
    embedder = OpenAIEmbedder(client=StubOpenAI())
    assert embedder.uncached(["a", "b"]) == ["a", "b"]  # no cache: everything
    embedder.cache = EmbeddingCache(tmp_path / "e.sqlite")
    embedder.embed(["a"])
    assert embedder.uncached(["a", "b", "b"]) == ["b"]


def test_a_text_repeated_in_one_call_is_embedded_once(tmp_path):
    client = StubOpenAI()
    embedder = OpenAIEmbedder(client=client)
    embedder.cache = EmbeddingCache(tmp_path / "e.sqlite")
    embedded = embedder.embed(["a", "b", "a"])
    assert client.calls == [["a", "b"]]
    assert embedded.vectors[0].tolist() == embedded.vectors[2].tolist()


def test_the_cache_holds_full_vectors_so_another_length_is_free(tmp_path):
    cache = EmbeddingCache(tmp_path / "e.sqlite")
    full = OpenAIEmbedder(client=StubOpenAI())
    full.cache = cache
    full.embed(["a", "b"])  # the stub returns [0, 1] and [1, 1]

    short = OpenAIEmbedder(dimensions=1, client=StubOpenAI())
    short.cache = cache
    embedded = short.embed(["a", "b"])
    assert short.client.calls == [] and embedded.tokens == 0
    assert embedded.vectors.tolist() == [[0.0], [1.0]]  # cut to one dimension, rescaled


def test_openai_dimensions_truncates_without_a_cache():
    embedded = OpenAIEmbedder(dimensions=1, client=StubOpenAI()).embed(["a", "b"])
    assert embedded.vectors.shape == (2, 1)


def test_different_models_do_not_share_cached_vectors(tmp_path):
    cache = EmbeddingCache(tmp_path / "e.sqlite")
    small, large = (
        OpenAIEmbedder(model=m, client=StubOpenAI())
        for m in ("text-embedding-3-small", "text-embedding-3-large")
    )
    small.cache = large.cache = cache
    small.embed(["a"])
    large.embed(["a"])
    assert large.client.calls == [["a"]] and len(cache) == 2


# --- Local embedder ------------------------------------------------------------------------


class FakeEncoder:
    """Stands in for a SentenceTransformer. A text's vector is (its length, 1, 0), so the
    test can see what text reached the model, and the vectors are not length 1."""

    max_seq_length = 6

    def __init__(self):
        self.seen: list[list[str]] = []

    def tokenizer(self, texts, truncation):
        return {"input_ids": [text.split() for text in texts]}

    def encode(self, texts, **settings):
        assert settings["normalize_embeddings"] is False
        self.seen.append(list(texts))
        return np.array([[len(text), 1.0, 0.0] for text in texts])


def test_prefix_goes_on_the_side_it_is_configured_for():
    encoder = FakeEncoder()
    embedder = LocalEmbedder("fake", query_prefix="query: ", encoder=encoder)
    embedder.embed(["who won"], kind="query")
    embedder.embed(["the cup went to Spain"], kind="document")
    embedder.embed(["no kind given"])
    assert encoder.seen == [["query: who won"], ["the cup went to Spain"], ["no kind given"]]


def test_local_embedder_returns_raw_float32_vectors_and_bills_nothing():
    embedded = LocalEmbedder("fake", encoder=FakeEncoder()).embed(["abcd"])
    assert embedded.vectors.dtype == np.float32
    assert embedded.vectors.tolist() == [[4.0, 1.0, 0.0]]  # not normalised here
    assert embedded.tokens == 0


def test_local_dimensions_truncate_and_rescale():
    embedded = LocalEmbedder("fake", dimensions=2, encoder=FakeEncoder()).embed(["abc"])
    assert embedded.vectors[0].tolist() == pytest.approx([3 / 10**0.5, 1 / 10**0.5])


def test_inputs_longer_than_the_model_reads_are_counted():
    embedder = LocalEmbedder("fake", document_prefix="passage: ", encoder=FakeEncoder())
    embedder.embed(["one two three", "one two three four five six"])
    assert embedder.truncated_inputs == 1  # the prefix counts towards the limit


def test_query_and_document_vectors_are_cached_apart_only_when_the_prefix_differs(tmp_path):
    cache = EmbeddingCache(tmp_path / "e.sqlite")
    prefixed = LocalEmbedder("fake", query_prefix="query: ", encoder=FakeEncoder())
    prefixed.cache = cache
    as_document = prefixed.embed(["same text"], kind="document").vectors
    as_query = prefixed.embed(["same text"], kind="query").vectors
    assert as_query.tolist() != as_document.tolist()
    assert len(cache) == 2

    # With no prefix the two kinds are one model input, so one cached vector serves both.
    plain = LocalEmbedder("fake", encoder=FakeEncoder())
    plain.cache = cache
    plain.embed(["same text"], kind="query")
    assert plain._encoder.seen == [] and len(cache) == 2

    # Another revision of the weights is another model.
    other = LocalEmbedder("fake", revision="abc123", encoder=FakeEncoder())
    other.cache = cache
    other.embed(["same text"])
    assert len(cache) == 3


def test_building_a_local_embedder_loads_nothing():
    embedder = LocalEmbedder("BAAI/bge-small-en-v1.5")
    assert embedder._encoder is None and embedder.model == "BAAI/bge-small-en-v1.5"


# --- In the pipeline -----------------------------------------------------------------------


@register("embedder", "recording")
class RecordingEmbedder:
    """Remembers which kind each call asked for."""

    model = "recording"
    kinds: list[str] = []

    def embed(self, texts: list[str], kind: str = "document") -> Embedded:
        RecordingEmbedder.kinds.append(kind)
        return Embedded(np.ones((len(texts), 2), dtype=np.float32), tokens=0)


def test_pipeline_embeds_chunks_as_documents_and_the_question_as_a_query(tmp_path):
    RecordingEmbedder.kinds = []
    pipeline = Pipeline(offline_config(tmp_path, embedder={"name": "recording"}))
    pipeline.ingest([Document("d", "The ferry was late.")])
    pipeline.retrieve("Why was the ferry late?")
    assert RecordingEmbedder.kinds == ["document", "query"]


def test_adding_a_cache_to_a_config_does_not_invalidate_its_index(tmp_path):
    Pipeline(offline_config(tmp_path)).ingest([Document("d", "The ferry was late.")])
    cached = offline_config(tmp_path, embedding_cache=tmp_path / "e.sqlite")
    assert Pipeline(cached).load() is True


def test_pipeline_gives_the_embedder_its_cache_and_repeats_are_free(tmp_path):
    cfg = offline_config(
        tmp_path,
        embedder={"name": "openai", "params": {"model": "text-embedding-3-small"}},
        embedding_cache=tmp_path / "e.sqlite",
    )
    first = Pipeline(cfg)
    first.embedder._client = StubOpenAI()
    first.ingest([Document("d", "The ferry was late.")])
    asked = first.retrieve("ferry?")
    assert asked.usage["embed_tokens"] == 3

    second = Pipeline(cfg)
    second.embedder._client = StubOpenAI()
    assert second.load() is True
    again = second.retrieve("ferry?")
    assert again.usage["embed_tokens"] == 0 and second.embedder.client.calls == []
    assert again.candidates[0].score == asked.candidates[0].score
