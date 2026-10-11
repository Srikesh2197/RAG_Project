"""Stage 6: the reranker slot, the cross-encoder's pairing, batching and cache, and the
listwise prompt and its parsing. No model and no network: the cross-encoder is given a
stand-in scorer and the listwise reranker a scripted LLM."""

import json
import re
from pathlib import Path

import pytest

from ragbasics.config import EvalConfig, PipelineConfig, load_config
from ragbasics.eval.dataset import write_jsonl
from ragbasics.eval.runner import ConfirmationNeeded, EstimateOnly, run_eval
from ragbasics.generation.llm import Generation
from ragbasics.pipeline import Pipeline
from ragbasics.registry import register
from ragbasics.rerank.base import passage_text, reorder
from ragbasics.rerank.cache import RerankCache, pair_key
from ragbasics.rerank.cross_encoder import CrossEncoderReranker
from ragbasics.rerank.llm_listwise import (
    SYSTEM_PROMPT,
    ListwiseReranker,
    build_prompt,
    parse_ranking,
)
from ragbasics.types import Candidate, Chunk
from test_eval_runner import DOCS as EVAL_DOCS
from test_eval_runner import QUESTIONS
from test_pipeline import DOCS, offline_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def candidates(*texts: str) -> list[Candidate]:
    return [
        Candidate(Chunk(f"c{i}", "doc", text, 0, len(text)), 1.0 / i, i, "hybrid")
        for i, text in enumerate(texts, start=1)
    ]


def ids(found: list[Candidate]) -> list[str]:
    return [c.chunk.chunk_id for c in found]


# --- Sorting -------------------------------------------------------------------------------


def test_reorder_sorts_by_score_renumbers_and_leaves_ties_in_first_stage_order():
    found = candidates("a", "b", "c", "d")
    ranked = reorder(found, [0.1, 0.7, 0.1, 0.9], "rerank")
    assert ids(ranked) == ["c4", "c2", "c1", "c3"]  # c1 and c3 tie; c1 was ahead
    assert [c.rank for c in ranked] == [1, 2, 3, 4]
    assert [c.score for c in ranked] == [0.9, 0.7, 0.1, 0.1]
    assert {c.retriever for c in ranked} == {"rerank"}
    # A reranker with nothing to say changes nothing.
    assert ids(reorder(found, [0.0] * 4, "rerank")) == ids(found)


# --- Cross-encoder -------------------------------------------------------------------------


class WordOverlapScorer:
    """Stands in for the model: the score is the number of question words in the passage,
    a "token" is a word, and every batch it is given is recorded."""

    def __init__(self, max_length: int = 512, identity: str = "overlap@v1"):
        self.max_length = max_length
        self.identity = identity
        self.batches: list[list[tuple[str, str]]] = []

    def lengths(self, pairs):
        return [len(q.split()) + len(p.split()) for q, p in pairs]

    def forward(self, pairs):
        self.batches.append(list(pairs))
        return [float(len(set(q.lower().split()) & set(p.lower().split()))) for q, p in pairs]


def test_cross_encoder_scores_each_candidate_against_the_question_and_sorts():
    scorer = WordOverlapScorer()
    reranker = CrossEncoderReranker("stub", scorer=scorer)
    found = candidates("the bakery sold bread", "fog delayed the ferry", "the ferry left")
    result = reranker.rerank("what delayed the ferry", found)
    assert ids(result.candidates) == ["c2", "c3", "c1"]
    assert [c.score for c in result.candidates] == [3.0, 2.0, 1.0]
    # Every pair carries the question.
    assert {q for batch in scorer.batches for q, _ in batch} == {"what delayed the ferry"}
    assert set(result.timings) == {"rerank"}
    assert result.cached is False and result.notes == {"truncated_pairs": 0}


def test_cross_encoder_batches_shortest_first_and_puts_scores_back_in_order():
    scorer = WordOverlapScorer()
    reranker = CrossEncoderReranker("stub", scorer=scorer, batch_size=2)
    passages = ["a b c d e f", "a", "a b c d", "a b", "a b c"]
    scores, computed = reranker.scores([("a c", p) for p in passages])
    assert computed == 5
    assert [[p for _, p in batch] for batch in scorer.batches] == [
        ["a", "a b"], ["a b c", "a b c d"], ["a b c d e f"],
    ]
    assert scores == [2.0, 1.0, 2.0, 1.0, 2.0]


def test_cross_encoder_counts_pairs_longer_than_the_model_reads():
    reranker = CrossEncoderReranker("stub", scorer=WordOverlapScorer(max_length=5))
    result = reranker.rerank("one two", candidates("a b c", "a b c d", "a b c d e f"))
    assert result.notes == {"truncated_pairs": 2}
    assert reranker.truncated_pairs == 2


def test_cached_scores_are_not_computed_again_and_come_back_identical(tmp_path):
    scorer = WordOverlapScorer()
    path = tmp_path / "cache" / "rerank.sqlite"
    reranker = CrossEncoderReranker("stub", scorer=scorer, cache=path)
    assert not path.exists()  # opened on first use
    found = candidates("fog delayed the ferry", "the ferry left", "the ferry left")
    first = reranker.rerank("what delayed the ferry", found)
    assert sum(len(batch) for batch in scorer.batches) == 2  # the repeated passage once
    assert len(RerankCache(path)) == 2

    # A new reranker over the same file computes nothing. One pass over a long list
    # therefore also serves every shorter depth.
    again_scorer = WordOverlapScorer()
    again = CrossEncoderReranker("stub", scorer=again_scorer, cache=path)
    second = again.rerank("what delayed the ferry", found[:2])
    assert again_scorer.batches == [] and second.cached is True
    assert [c.score for c in second.candidates] == [c.score for c in first.candidates[:2]]

    # Another model, or another question, is another key.
    other = WordOverlapScorer(identity="overlap@v2")
    CrossEncoderReranker("stub", scorer=other, cache=path).rerank("what delayed the ferry", found)
    assert sum(len(batch) for batch in other.batches) == 2
    assert pair_key("m", "ab", "c") != pair_key("m", "a", "bc")


def test_header_changes_what_the_reranker_reads_and_not_the_chunk():
    metadata = {"title": "Ferry woes", "source": "Port News", "published_at": "2023-10-07"}
    chunk = Chunk("c1", "doc", "The ferry was late.", 0, 19, metadata=metadata)
    assert passage_text(chunk, header=False) == "The ferry was late."
    assert passage_text(chunk, header=True) == (
        "Ferry woes. Port News, October 7, 2023.\nThe ferry was late."
    )
    scorer = WordOverlapScorer()
    reranker = CrossEncoderReranker("stub", scorer=scorer, header=True)
    result = reranker.rerank("port october", [Candidate(chunk, 1.0, 1, "hybrid")])
    assert result.candidates[0].score == 2.0
    assert result.candidates[0].chunk.text == "The ferry was late."


def test_cross_encoder_refuses_settings_it_does_not_know():
    with pytest.raises(ValueError, match="scoring"):
        CrossEncoderReranker("stub", scoring="colbert")
    with pytest.raises(ValueError, match="depth"):
        CrossEncoderReranker("stub", depth=0)


# --- Listwise LLM --------------------------------------------------------------------------


def test_parse_ranking_reads_a_clean_ordering():
    assert parse_ranking("3 > 1 > 2", 3) == ([2, 0, 1], {"invalid": 0, "repeated": 0, "missing": 0})
    assert parse_ranking("[2] > [3] > [1]", 3)[0] == [1, 2, 0]


def test_parse_ranking_repairs_repeats_gaps_and_numbers_out_of_range():
    order, repairs = parse_ranking("4 > 2 > 2 > 9 > 0", 5)
    # 4 and 2 were ranked; 1, 3 and 5 were never mentioned and keep their order below.
    assert order == [3, 1, 0, 2, 4]
    assert repairs == {"invalid": 2, "repeated": 1, "missing": 3}
    # A reply with no numbers leaves the list as it was.
    assert parse_ranking("I cannot rank these.", 3) == (
        [0, 1, 2], {"invalid": 0, "repeated": 0, "missing": 3},
    )


def test_listwise_prompt_numbers_the_passages_and_ends_with_the_question():
    assert build_prompt("Who?", ["First.\n", "Second."]) == (
        "[1]\nFirst.\n\n[2]\nSecond.\n\nQuestion: Who?"
    )
    assert "exactly once" in SYSTEM_PROMPT


@register("generator", "reverse_ranker")
class ReverseRanker:
    """Stands in for the LLM: ranks the passages in reverse order. Priced as a real
    model, so cost estimates are not zero."""

    model = "gpt-6-luna"
    calls = 0

    def generate(self, system: str, user: str) -> Generation:
        type(self).calls += 1
        n = len(re.findall(r"^\[\d+\]$", user, flags=re.MULTILINE))
        reply = " > ".join(str(i) for i in range(n, 0, -1))
        return Generation(reply, input_tokens=1000, output_tokens=20, stop_reason="stop")


def test_listwise_reranker_applies_the_models_ordering_and_reports_tokens(tmp_path):
    reranker = ListwiseReranker({"name": "reverse_ranker"}, cache=tmp_path / "llm.sqlite")
    found = candidates("a", "b", "c")
    result = reranker.rerank("q", found)
    assert ids(result.candidates) == ["c3", "c2", "c1"]
    assert [c.score for c in result.candidates] == [1.0, 0.5, pytest.approx(1 / 3)]
    assert (result.input_tokens, result.output_tokens, result.cached) == (1000, 20, False)
    assert result.notes == {"invalid": 0, "repeated": 0, "missing": 0, "stop_reason": "stop"}
    # The same prompt again comes from the cache.
    assert reranker.is_cached("q", found) and not reranker.is_cached("q", found[:2])
    assert reranker.rerank("q", found).cached is True
    assert ListwiseReranker({"name": "reverse_ranker"}).rerank("q", []).candidates == []


def test_a_prompt_the_provider_refuses_keeps_the_first_stage_order():
    class BadRequestError(Exception):
        pass

    class Refuses:
        model = "gpt-6-luna"

        def __init__(self, error):
            self.error = error

        def generate(self, system, user):
            raise self.error

    found = candidates("a", "b", "c")
    result = ListwiseReranker(Refuses(BadRequestError("flagged"))).rerank("q", found)
    assert ids(result.candidates) == ids(found)
    assert result.notes == {"refused": True} and result.input_tokens == 0
    # Anything else, such as a missing key or a timeout, is not swallowed.
    with pytest.raises(RuntimeError):
        ListwiseReranker(Refuses(RuntimeError("no key"))).rerank("q", found)


# --- In the pipeline -----------------------------------------------------------------------

LISTWISE = {"name": "llm_listwise", "params": {"generator": {"name": "reverse_ranker"}}}


def listwise(depth: int) -> dict:
    return {"name": "llm_listwise", "params": LISTWISE["params"] | {"depth": depth}}


def test_there_is_no_reranker_by_default_so_old_configs_are_unchanged(tmp_path):
    cfg = offline_config(tmp_path)
    assert cfg.reranker is None
    pipeline = Pipeline(cfg)
    pipeline.ingest(DOCS)
    trace = pipeline.ask("Why was the ferry delayed?")
    assert trace.retrieved == [] and trace.rerank == {}
    assert set(trace.timings) == {"embed_query", "search", "generate"}
    assert "reranker" not in trace.models


def test_a_depth_at_or_below_k_reorders_the_top_and_leaves_the_rest(tmp_path):
    plain = Pipeline(offline_config(tmp_path, top_k=3))
    plain.ingest(DOCS)
    before = ids(plain.retrieve("Why was the ferry delayed?").candidates)

    pipeline = Pipeline(offline_config(tmp_path, top_k=3, reranker=listwise(2)))
    assert pipeline.load() is True  # the reranker is not part of what the index depends on
    trace = pipeline.retrieve("Why was the ferry delayed?")
    assert ids(trace.retrieved) == before
    assert ids(trace.candidates) == [before[1], before[0], before[2]]
    assert [c.rank for c in trace.candidates] == [1, 2, 3]
    assert [c.retriever for c in trace.candidates] == ["rerank", "rerank", "dense"]
    assert trace.candidates[2].score == trace.retrieved[2].score  # untouched below the depth
    assert trace.models["reranker"] == "gpt-6-luna"
    assert trace.rerank["depth"] == 2 and trace.rerank["cached"] is False
    assert "rerank" in trace.timings
    assert trace.usage["rerank_input_tokens"] == 1000
    assert trace.cost_usd == pytest.approx((1000 * 0.10 + 20 * 0.50) / 1e6)


def test_a_depth_above_k_can_bring_in_chunks_the_retriever_ranked_below_k(tmp_path):
    plain = Pipeline(offline_config(tmp_path, top_k=3))
    plain.ingest(DOCS)
    before = ids(plain.retrieve("Why was the ferry delayed?").candidates)

    pipeline = Pipeline(offline_config(tmp_path, top_k=1, reranker=listwise(3)))
    pipeline.load()
    trace = pipeline.ask("Why was the ferry delayed?")
    assert ids(trace.retrieved) == before  # fetched as deep as the reranker reads
    assert ids(trace.candidates) == [before[2]]  # the retriever's third is now first
    assert [c.chunk for c in trace.candidates] == trace.context


def test_reranking_works_behind_hybrid_retrieval_and_keeps_each_retrievers_list(tmp_path):
    cfg = offline_config(tmp_path, top_k=2, retriever={"name": "hybrid"}, reranker=listwise(2))
    pipeline = Pipeline(cfg)
    pipeline.ingest(DOCS)
    trace = pipeline.retrieve("What closed the airport?")
    assert set(trace.retrievers) == {"dense", "sparse"}
    assert {c.retriever for c in trace.retrieved} == {"hybrid"}
    assert ids(trace.candidates) == ids(trace.retrieved)[::-1]


# --- In the evaluator ----------------------------------------------------------------------


@pytest.fixture
def evaluation(tmp_path):
    write_jsonl(tmp_path / "documents.jsonl", EVAL_DOCS)
    write_jsonl(tmp_path / "questions_dev.jsonl", QUESTIONS)
    settings = {
        "name": "tiny",
        "index_dir": tmp_path / "index",
        "chunker": {"name": "fixed", "params": {"size": 64}},
        "embedder": {"name": "hashing", "params": {"dimensions": 256}},
        "store": {"name": "numpy"},
        "top_k": 2,
        "generator": {"name": "scripted"},
    }
    plain = PipelineConfig.model_validate(settings)
    reranked = PipelineConfig.model_validate(
        settings | {
            "name": "tiny-rerank",
            "reranker": {
                "name": "llm_listwise",
                "params": {
                    "generator": {"name": "reverse_ranker"},
                    "depth": 4,
                    "cache": str(tmp_path / "cache.sqlite"),
                },
            },
        }
    )
    eval_cfg = EvalConfig.model_validate(
        {
            "questions": tmp_path / "questions_dev.jsonl",
            "documents": tmp_path / "documents.jsonl",
            "depth": 4,
            "ks": [1, 2],
            "rank_k": 10,
            "token_budget": 15,
            "judge": {"name": "openai", "params": {"model": "unused"}},
            "bootstrap": {"samples": 200, "seed": 0},
            "workers": 2,
            "cache": tmp_path / "cache.sqlite",
            "confirm_above_usd": 1.0,
        }
    )
    Pipeline(plain).ingest(EVAL_DOCS)

    def run(cfg, eval_cfg=eval_cfg, **kwargs):
        return run_eval(
            cfg, eval_cfg, answers=False, runs_dir=tmp_path / "runs",
            ledger=tmp_path / "ledger.csv", log=lambda _: None, **kwargs,
        )

    return run, plain, reranked, eval_cfg, tmp_path


def rows(run_dir: Path) -> dict:
    with open(run_dir / "questions.jsonl") as f:
        return {row["question_id"]: row for row in map(json.loads, f)}


def test_a_reranked_run_records_the_list_before_and_after(evaluation):
    run, plain, reranked, _, tmp_path = evaluation
    before, after = rows(run(plain)), rows(run(reranked))
    for qid, row in after.items():
        first_stage = [chunk_id for chunk_id, _ in before[qid]["retrieved"]]
        assert [chunk_id for chunk_id, _ in row["before_rerank"]] == first_stage
        assert [chunk_id for chunk_id, _ in row["retrieved"]] == first_stage[::-1]
        assert row["rerank"]["depth"] == 4
    assert "before_rerank" not in before["q1"]
    # The gold chunk for q1 was first and is now last of four.
    assert before["q1"]["gold_ranks"] == [1] and after["q1"]["gold_ranks_before"] == [1]
    assert after["q1"]["gold_ranks"] == [4]
    summary = json.loads((tmp_path / "runs" / "tiny-rerank" / "metrics.json").read_text())
    assert summary["models"]["reranker"] == "gpt-6-luna"
    # Five calls of 1,000 tokens in and 20 out, billed once and written to the ledger.
    assert summary["cost_usd"]["billed_this_run"] == pytest.approx(5 * 0.00011, abs=1e-4)
    assert "gpt-6-luna" in (tmp_path / "ledger.csv").read_text()


def test_an_llm_reranker_is_estimated_before_any_call_and_free_when_cached(evaluation):
    run, _, reranked, eval_cfg, tmp_path = evaluation
    ReverseRanker.calls = 0
    with pytest.raises(EstimateOnly):
        run(reranked, estimate_only=True)
    with pytest.raises(ConfirmationNeeded):
        run(reranked, eval_cfg.model_copy(update={"confirm_above_usd": 0.0}))
    assert ReverseRanker.calls == 0 and not (tmp_path / "runs").exists()

    run(reranked)
    assert ReverseRanker.calls == len(QUESTIONS)
    # Every ordering is now cached: a repeat calls nothing, bills nothing, and an
    # estimate-only run has nothing to stop for.
    lines = len((tmp_path / "ledger.csv").read_text().splitlines())
    run(reranked, estimate_only=True)
    assert ReverseRanker.calls == len(QUESTIONS)
    assert len((tmp_path / "ledger.csv").read_text().splitlines()) == lines


# --- Configs -------------------------------------------------------------------------------


def test_stage06_configs_change_only_the_reranker():
    best = load_config(REPO_ROOT / "configs/stage05_retrieval/hybrid-rrf.yaml", PipelineConfig)
    assert best.reranker is None
    paths = sorted((REPO_ROOT / "configs" / "stage06_rerank").glob("*.yaml"))
    assert paths
    for path in paths:
        cfg = load_config(path, PipelineConfig)
        Pipeline(cfg.model_copy(update={"embedding_cache": None}))  # loads no model
        assert cfg.name == f"rerank-{path.stem}"
        assert cfg.reranker is not None and cfg.reranker.params["depth"] in (20, 50, 100)
        assert f"-{cfg.reranker.params['depth']}" in path.stem
        same = {"index_dir", "chunker", "embedder", "store", "top_k", "generator", "overfetch"}
        # Rows that also change the retriever say so in their file name.
        if not any(tag in path.stem for tag in ("weighted", "bm25-header", "qwen3-embed")):
            same.add("retriever")
        if "qwen3-embed" in path.stem:
            same -= {"index_dir", "embedder"}
        if path.stem.endswith("-top25"):  # the answer-quality config also sets top_k
            same.remove("top_k")
        assert cfg.model_dump(include=same) == best.model_dump(include=same)
