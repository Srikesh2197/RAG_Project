"""Stage 5: the tokeniser, BM25 against a hand-computed example, the two fusion methods,
and the retriever slot in the pipeline. No model and no network."""

import math
from pathlib import Path

import pytest

from ragbasics.config import PipelineConfig, load_config
from ragbasics.index.bm25 import BM25Index
from ragbasics.pipeline import Pipeline
from ragbasics.retrieval.fusion import normalise, reciprocal_rank_fusion, weighted_fusion
from ragbasics.retrieval.hybrid import HybridRetriever
from ragbasics.retrieval.sparse import SparseRetriever, header
from ragbasics.retrieval.tokenize import Tokenizer, porter_stem
from ragbasics.types import Candidate, Chunk, Document
from test_pipeline import DOCS, offline_config

REPO_ROOT = Path(__file__).resolve().parents[1]

# --- Tokeniser -----------------------------------------------------------------------------


def test_tokeniser_lowercases_and_splits_on_anything_but_letters_and_digits():
    assert Tokenizer()("Amazon's GPT-4 deal: $1.2 billion in the U.S.") == [
        "amazon", "gpt", "4", "deal", "1", "2", "billion", "in", "the", "u", "s",
    ]


def test_stopwords_and_stemming_are_options():
    text = "The companies were acquiring startups"
    assert Tokenizer(stopwords=True)(text) == ["companies", "acquiring", "startups"]
    assert Tokenizer(stemmer="porter")(text) == ["the", "compani", "were", "acquir", "startup"]
    with pytest.raises(ValueError, match="stemmer"):
        Tokenizer(stemmer="snowball")


def test_porter_stemmer_on_known_words():
    expected = {
        "caresses": "caress", "ponies": "poni", "cats": "cat", "agreed": "agre",
        "plastered": "plaster", "motoring": "motor", "hopping": "hop", "filing": "file",
        "falling": "fall", "happy": "happi", "sky": "sky", "relational": "relat",
        "conditional": "condit", "generalization": "gener", "electricity": "electr",
        "adjustment": "adjust", "probate": "probat", "rate": "rate", "bled": "bled",
        "controlling": "control", "hopefulness": "hope", "adoption": "adopt", "roll": "roll",
    }
    assert {word: porter_stem(word) for word in expected} == expected


def test_stemming_joins_forms_of_a_word_and_also_words_that_are_not_related():
    assert porter_stem("acquired") == porter_stem("acquiring") == "acquir"
    assert porter_stem("university") == porter_stem("universe") == "univers"


# --- BM25 ----------------------------------------------------------------------------------

# Lengths 2, 4 and 3: the average is 3.
CORPUS = [
    ["ferry", "fog"],
    ["ferry", "ferry", "late", "harbour"],
    ["bakery", "revenue", "rose"],
]


def test_bm25_matches_a_hand_computed_example():
    index = BM25Index.build(CORPUS)
    # "ferry" is in 2 of 3 chunks: idf = ln(1 + (3 - 2 + 0.5) / (2 + 0.5)) = ln(1.6).
    assert index.document_frequency("ferry") == 2
    assert index.idf("ferry") == pytest.approx(math.log(1.6))
    # k1 = 1.2, b = 0.75. Chunk 0 (length 2, one "ferry"): 1.2 * (0.25 + 0.75 * 2/3) = 0.9,
    # so 1 / (1 + 0.9). Chunk 1 (length 4, two): 1.2 * (0.25 + 0.75 * 4/3) = 1.5, so 2 / 3.5.
    scores = index.scores(["ferry"], k1=1.2, b=0.75)
    assert scores.tolist() == pytest.approx(
        [math.log(1.6) / 1.9, math.log(1.6) * 2 / 3.5, 0.0], rel=1e-6
    )
    assert [i for i, _ in index.search(["ferry"], k=3)] == [1, 0]  # chunk 2 never matched


def test_k1_sets_how_much_a_repeated_term_adds():
    index = BM25Index.build(CORPUS)
    # At k1 = 0 only presence counts: one "ferry" scores the same as two.
    presence = index.scores(["ferry"], k1=0, b=0)
    assert presence[0] == pytest.approx(presence[1])
    # The gain from the second occurrence grows with k1 and never reaches double.
    low, high = index.scores(["ferry"], k1=0.5, b=0), index.scores(["ferry"], k1=5, b=0)
    assert 1 < low[1] / low[0] < high[1] / high[0] < 2


def test_b_sets_how_much_a_long_chunk_is_marked_down():
    # The same single "fog", in a short chunk and in a long one.
    index = BM25Index.build([["fog", "ferry"], ["fog", "a", "b", "c", "d", "e"]])
    ignored = index.scores(["fog"], b=0)
    assert ignored[0] == pytest.approx(ignored[1])
    corrected = index.scores(["fog"], b=1)
    assert corrected[0] > corrected[1]


def test_a_rare_term_outweighs_a_common_one():
    index = BM25Index.build([["the", "fog"], ["the", "sun"], ["the", "rain"], ["the", "the"]])
    assert index.idf("fog") > 10 * index.idf("the")
    assert index.search(["the", "fog"], k=1)[0][0] == 0


def test_bm25_ignores_unknown_terms_and_counts_a_repeated_query_term_once():
    index = BM25Index.build(CORPUS)
    assert index.search(["zebra"], k=3) == []
    once, twice = index.scores(["ferry"]), index.scores(["ferry", "zebra", "ferry"])
    assert once.tolist() == twice.tolist()


def test_bm25_keeps_chunk_order_for_equal_scores():
    index = BM25Index.build([["fog"], ["sun"], ["fog"]])
    assert [i for i, _ in index.search(["fog"], k=5)] == [0, 2]


def test_bm25_index_round_trips_through_disk(tmp_path):
    index = BM25Index.build(CORPUS)
    index.save(tmp_path / "bm25", {"stemmer": "none"})
    loaded, meta = BM25Index.load(tmp_path / "bm25")
    assert meta == {"stemmer": "none"}
    assert loaded.search(["ferry", "late"], k=3) == index.search(["ferry", "late"], k=3)


# --- Fusion --------------------------------------------------------------------------------


def ranked(retriever: str, scored: list[tuple[str, float]]) -> list[Candidate]:
    return [
        Candidate(Chunk(name, "doc", name, 0, 1), score, rank, retriever)
        for rank, (name, score) in enumerate(scored, start=1)
    ]


def ids(candidates: list[Candidate]) -> list[str]:
    return [c.chunk.chunk_id for c in candidates]


def test_reciprocal_rank_fusion_matches_a_hand_computed_example():
    lists = {
        "dense": ranked("dense", [("A", 0.9), ("B", 0.8), ("C", 0.7)]),
        "sparse": ranked("sparse", [("C", 30.0), ("A", 9.0), ("D", 1.0)]),
    }
    fused = reciprocal_rank_fusion(lists, k=60)
    # A: 1/61 + 1/62. C: 1/63 + 1/61. B: 1/62. D: 1/63.
    assert ids(fused) == ["A", "C", "B", "D"]
    assert fused[0].score == pytest.approx(1 / 61 + 1 / 62)
    assert [c.rank for c in fused] == [1, 2, 3, 4]
    assert {c.retriever for c in fused} == {"hybrid"}
    # The scores were never read: C's BM25 lead of 30 against 9 changed nothing.


def test_rrf_constant_decides_between_one_first_place_and_two_third_places():
    lists = {
        "dense": ranked("dense", [("A", 0.9), ("X", 0.8), ("B", 0.7)]),
        "sparse": ranked("sparse", [("C", 9.0), ("Y", 8.0), ("B", 7.0)]),
    }
    # B: 2 / (k + 3). A: 1 / (k + 1). B is ahead once k is above 1.
    assert ids(reciprocal_rank_fusion(lists, k=0.5))[0] == "A"
    assert ids(reciprocal_rank_fusion(lists, k=60))[0] == "B"


def test_normalise_by_hand():
    assert normalise([10.0, 4.0, 2.0], "minmax") == pytest.approx([1.0, 0.25, 0.0])
    # Mean 4, standard deviation sqrt(8/3).
    spread = math.sqrt(8 / 3)
    assert normalise([6.0, 4.0, 2.0], "zscore") == pytest.approx([2 / spread, 0.0, -2 / spread])
    assert normalise([3.0, 3.0], "minmax") == [0.0, 0.0]
    assert normalise([], "zscore") == []
    with pytest.raises(ValueError, match="normaliser"):
        normalise([1.0], "rank")


def test_adding_raw_scores_lets_bm25_decide_and_normalising_does_not():
    dense = ranked("dense", [("A", 0.9), ("B", 0.5), ("C", 0.1)])
    sparse = ranked("sparse", [("B", 12.0), ("A", 11.0), ("C", 2.0)])
    raw = {c.chunk.chunk_id: c.score for c in dense}
    for c in sparse:
        raw[c.chunk.chunk_id] += c.score
    # Dense prefers A by a wide margin and BM25 prefers B narrowly. The raw sum follows
    # BM25, because its numbers are ten times larger.
    assert max(raw, key=raw.get) == "B"
    # After min-max: A = 0.5 * 1 + 0.5 * 0.9, B = 0.5 * 0.5 + 0.5 * 1.
    fused = weighted_fusion(
        {"dense": dense, "sparse": sparse}, {"dense": 0.5, "sparse": 0.5}, "minmax"
    )
    assert ids(fused) == ["A", "B", "C"]
    assert [c.score for c in fused] == pytest.approx([0.95, 0.75, 0.0])


def test_weighted_fusion_at_weight_one_is_the_dense_order_and_missing_chunks_sit_last():
    lists = {
        "dense": ranked("dense", [("A", 0.9), ("B", 0.5), ("C", 0.1)]),
        "sparse": ranked("sparse", [("D", 12.0), ("B", 11.0)]),
    }
    for method in ("minmax", "zscore"):
        assert ids(weighted_fusion(lists, {"dense": 1.0, "sparse": 0.0}, method))[:3] == [
            "A", "B", "C",
        ]
    # D is missing from the dense list, so it gets that list's lowest score, 0.
    fused = weighted_fusion(lists, {"dense": 0.5, "sparse": 0.5}, "minmax")
    assert {c.chunk.chunk_id: c.score for c in fused}["D"] == pytest.approx(0.5)


def test_hybrid_retriever_refuses_settings_it_does_not_know():
    HybridRetriever(fusion={"method": "weighted", "weight": 0.7, "normalise": "zscore"})
    for fusion in (
        {"method": "borda"},
        {"method": "rrf", "weight": 0.5},
        {"method": "weighted", "weight": 1.5},
        {"method": "weighted", "weight": 0.5, "normalise": "rank"},
    ):
        with pytest.raises(ValueError):
            HybridRetriever(fusion=fusion)


# --- In the pipeline -----------------------------------------------------------------------


def test_the_retriever_defaults_to_dense_so_old_configs_are_unchanged(tmp_path):
    cfg = offline_config(tmp_path)
    assert cfg.retriever.name == "dense"
    pipeline = Pipeline(cfg)
    pipeline.ingest(DOCS)
    trace = pipeline.retrieve("Why was the ferry delayed?")
    assert {c.retriever for c in trace.candidates} == {"dense"}
    assert trace.retrievers == {"dense": trace.candidates}
    assert not (tmp_path / "index" / "bm25").exists()


def test_sparse_retrieval_ranks_by_shared_terms_and_embeds_nothing(tmp_path):
    cfg = offline_config(tmp_path, retriever={"name": "sparse"}, top_k=3)
    pipeline = Pipeline(cfg)
    pipeline.ingest(DOCS)
    trace = pipeline.retrieve("What closed the airport?")
    # All three articles contain "the", so all three match; the rare terms decide.
    assert [c.chunk.doc_id for c in trace.candidates][0] == "doc_c"
    assert trace.candidates[0].score > 5 * trace.candidates[1].score
    assert {c.retriever for c in trace.candidates} == {"sparse"}
    assert set(trace.timings) == {"sparse_search"}
    assert trace.usage == {"embed_tokens": 0} and trace.cost_usd == 0.0

    # With stopwords removed, the articles that shared only "the" are not returned at all.
    strict = offline_config(
        tmp_path, retriever={"name": "sparse", "params": {"stopwords": True}}, top_k=3
    )
    pipeline = Pipeline(strict)
    pipeline.load()
    assert [c.chunk.doc_id for c in pipeline.retrieve("What closed the airport?").candidates] == [
        "doc_c"
    ]


def test_hybrid_trace_keeps_each_retrievers_list_next_to_the_fused_one(tmp_path):
    cfg = offline_config(
        tmp_path,
        retriever={"name": "hybrid", "params": {"fusion": {"method": "rrf", "k": 60}}},
        top_k=2,
    )
    pipeline = Pipeline(cfg)
    pipeline.ingest(DOCS)
    trace = pipeline.ask("What closed the airport?")
    assert set(trace.retrievers) == {"dense", "sparse"}
    assert len(trace.retrievers["dense"]) == 3  # searched deeper than top_k before fusing
    assert {c.retriever for c in trace.retrievers["sparse"]} == {"sparse"}
    assert [c.rank for c in trace.candidates] == [1, 2]
    assert {c.retriever for c in trace.candidates} == {"hybrid"}
    assert trace.candidates[0].chunk.doc_id == "doc_c"  # first in both lists
    assert trace.candidates[0].score == pytest.approx(2 / 61)
    assert set(trace.timings) == {"embed_query", "search", "sparse_search", "fuse", "generate"}


def test_an_index_built_for_dense_search_loads_under_a_sparse_or_hybrid_config(tmp_path):
    Pipeline(offline_config(tmp_path)).ingest(DOCS)
    meta = (tmp_path / "index" / "meta.json").read_text()
    for name in ("sparse", "hybrid"):
        pipeline = Pipeline(offline_config(tmp_path, retriever={"name": name}))
        assert pipeline.load() is True
        assert pipeline.retrieve("ferry fog").candidates[0].chunk.doc_id == "doc_a"
    assert (tmp_path / "index" / "meta.json").read_text() == meta


def test_bm25_index_is_saved_per_tokeniser_setting_and_rebuilt_when_chunks_change(tmp_path):
    cfg = offline_config(tmp_path, retriever={"name": "sparse"})
    pipeline = Pipeline(cfg)
    pipeline.ingest(DOCS)
    pipeline.retrieve("ferry")
    saved = list((tmp_path / "index" / "bm25").iterdir())
    assert len(saved) == 1 and (saved[0] / "postings.npz").exists()

    # A second pipeline reads the saved index; k1 and b are not part of what it depends on.
    again = Pipeline(offline_config(tmp_path, retriever={"name": "sparse", "params": {"k1": 2}}))
    again.load()
    again.retrieve("ferry")
    assert list((tmp_path / "index" / "bm25").iterdir()) == saved

    # Another tokeniser setting is another index.
    other = Pipeline(
        offline_config(tmp_path, retriever={"name": "sparse", "params": {"stemmer": "porter"}})
    )
    other.load()
    assert other.retrieve("ferries delay").candidates[0].chunk.doc_id == "doc_a"
    assert len(list((tmp_path / "index" / "bm25").iterdir())) == 2

    # New chunks: the saved index no longer matches and is rebuilt.
    pipeline.ingest([Document("doc_e", "A zebra crossed the road.")])
    assert pipeline.retrieve("zebra").candidates[0].chunk.doc_id == "doc_e"


def test_header_puts_title_source_and_written_date_into_the_sparse_index_only(tmp_path):
    metadata = {
        "title": "Ferry woes", "source": "Port News", "published_at": "2023-10-07T08:00:00+00:00",
    }
    assert header(metadata) == "Ferry woes. Port News, October 7, 2023."
    assert header({}) == ""
    docs = [
        Document("doc_a", "The harbour ferry was delayed by fog.", metadata),
        Document("doc_b", "The harbour ferry was delayed by wind.", {"source": "Daily Wire"}),
    ]
    question = "What did Port News report on October 7 about the ferry?"
    plain = Pipeline(offline_config(tmp_path / "plain", retriever={"name": "sparse"}))
    plain.ingest(docs)
    # Without the header nothing in the text separates the two: the scores are equal.
    first, second = plain.retrieve(question).candidates
    assert first.score == pytest.approx(second.score)
    with_header = Pipeline(
        offline_config(tmp_path / "h", retriever={"name": "sparse", "params": {"header": True}})
    )
    with_header.ingest(docs)
    top = with_header.retrieve(question).candidates[0]
    assert top.chunk.doc_id == "doc_a"
    assert top.chunk.text == "The harbour ferry was delayed by fog."  # the text is unchanged


def test_sparse_search_over_parent_child_chunks_indexes_children_and_returns_parents(tmp_path):
    text = ("The harbour ferry was delayed by fog. The ferry left at noon. " * 4
            + "\n\nQuarterly revenue at the bakery rose. The bakery sold more bread. " * 4)
    cfg = offline_config(
        tmp_path,
        chunker={"name": "parent_child", "params": {"parent_size": 64, "child_size": 16}},
        retriever={"name": "sparse"},
        overfetch=4,
    )
    pipeline = Pipeline(cfg)
    pipeline.ingest([Document("doc_p", text)])
    trace = pipeline.retrieve("bakery bread", k=5)
    spans = [(c.chunk.start_char, c.chunk.end_char) for c in trace.candidates]
    assert len(spans) == len(set(spans))  # each parent once
    assert "bakery" in trace.candidates[0].chunk.text


def test_sparse_retriever_settings_leave_out_k1_and_b():
    assert SparseRetriever(k1=0.9, b=0.4, stemmer="porter").settings() == {
        "stopwords": False, "stemmer": "porter", "header": False,
    }


def test_stage05_configs_change_only_the_retriever():
    best = load_config(REPO_ROOT / "configs/stage03_chunking/recursive-128.yaml", PipelineConfig)
    paths = sorted((REPO_ROOT / "configs" / "stage05_retrieval").glob("*.yaml"))
    assert paths
    for path in paths:
        cfg = load_config(path, PipelineConfig)
        Pipeline(cfg.model_copy(update={"embedding_cache": None}))
        assert cfg.name == f"retr-{path.stem}"
        assert cfg.retriever.name in ("sparse", "hybrid")
        same = {"index_dir", "chunker", "embedder", "store", "top_k", "generator", "overfetch"}
        if path.stem.endswith("-top25"):  # the answer-quality config also sets top_k
            same.remove("top_k")
        assert cfg.model_dump(include=same) == best.model_dump(include=same)
