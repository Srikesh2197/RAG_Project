"""Stage 7: query transformations. The rule-based splitter, the parsing and repair of an
LLM's reply, the fallback, the combination of several lists, and the slot in the
pipeline and the evaluator. No model and no network: the LLM is scripted."""

import json
import threading
from pathlib import Path

import pytest

from ragbasics.config import PipelineConfig, load_config
from ragbasics.costs import Meter
from ragbasics.eval.runner import ConfirmationNeeded, EstimateOnly
from ragbasics.generation.llm import Generation
from ragbasics.pipeline import Pipeline
from ragbasics.query.base import Routing, route
from ragbasics.query.clauses import ClauseSplitter, split_clauses
from ragbasics.query.followup import SYSTEM_PROMPT as FOLLOWUP_SYSTEM
from ragbasics.query.followup import FollowUpRewriter, build_prompt
from ragbasics.query.llm import (
    Decomposer,
    HyDE,
    MultiQuery,
    Rewriter,
    clean_line,
    parse_lines,
    parse_passage,
)
from ragbasics.registry import register
from ragbasics.rerank.base import Reranked, merge, reorder
from ragbasics.rerank.cross_encoder import CrossEncoderReranker
from ragbasics.retrieval.fusion import round_robin
from ragbasics.types import Candidate, Chunk, Query
from test_pipeline import DOCS, offline_config
from test_rerank import WordOverlapScorer, candidates, evaluation, ids, rows  # noqa: F401

REPO_ROOT = Path(__file__).resolve().parents[1]

# --- The rule-based splitter ---------------------------------------------------------------

TWO_PARTS = "Did bakery revenue rise sharply and what closed the airport for good?"


def test_split_clauses_cuts_at_commas_and_joining_words():
    question = (
        "Does the TechCrunch article say sales rose in North America, while The Verge "
        "article reports criticism of electric vehicles?"
    )
    # The comma is the boundary, so the joining word stays on the second clause.
    assert split_clauses(question) == [
        "Does the TechCrunch article say sales rose in North America",
        "while The Verge article reports criticism of electric vehicles?",
    ]
    assert split_clauses(TWO_PARTS) == [
        "Did bakery revenue rise sharply", "what closed the airport for good?",
    ]


def test_split_clauses_glues_short_pieces_back_and_is_crude_about_lists():
    # ", 2023," is under five words and joins the piece before it, so the date stays whole.
    assert split_clauses("In 2023, did the harbour ferry company report a loss?") == [
        "In 2023, did the harbour ferry company report a loss?"
    ]
    assert split_clauses("Did revenue rise and who won?") == ["Did revenue rise and who won?"]
    assert split_clauses("Who won?") == ["Who won?"]
    # The rule knows no grammar: an "and" between two names is a boundary like any
    # other, and the second clause loses its subject. This is what the LLM
    # decomposition is compared against.
    question = (
        "After the report published on October 4, 2023, did Sporting News and Fortune "
        "both describe the betting lines as unchanged?"
    )
    assert split_clauses(question) == [
        "After the report published on October 4, 2023, did Sporting News",
        "Fortune both describe the betting lines as unchanged?",
    ]


def test_clause_splitter_needs_no_model_and_marks_an_unsplit_question_as_original():
    splitter = ClauseSplitter()
    result = splitter.transform(TWO_PARTS)
    assert result.queries == [
        Query("Did bakery revenue rise sharply", "clause"),
        Query("what closed the airport for good?", "clause"),
    ]
    assert (result.calls, result.input_tokens, result.cached) == (0, 0, True)
    assert splitter.model == "rule"
    assert splitter.transform("Who won?").queries == [Query("Who won?", "original")]
    capped = ClauseSplitter(max_queries=1).transform(TWO_PARTS)
    assert len(capped.queries) == 1 and capped.notes == {"extra": 1}


# --- Parsing a reply -----------------------------------------------------------------------


def test_clean_line_strips_numbering_bullets_labels_and_quotes():
    assert clean_line("  1. What delayed the ferry?  ") == "What delayed the ferry?"
    assert clean_line("- What delayed the ferry?") == "What delayed the ferry?"
    assert clean_line("2) “What delayed the ferry?”") == "What delayed the ferry?"
    assert clean_line("Rewritten query: ferry delay cause") == "ferry delay cause"
    assert clean_line("Sub-question 2: Who ran it?") == "Who ran it?"
    # A number that is part of the text is left alone.
    assert clean_line("2023 was the year of the ferry strike") == (
        "2023 was the year of the ferry strike"
    )
    assert clean_line("'s") == "'s"


def test_parse_lines_drops_blanks_and_repeats_and_cuts_at_the_limit():
    reply = "1. First one\n\n2. Second one\n3. first  ONE\n4. Third\n5. Fourth"
    assert parse_lines(reply, limit=3) == (
        ["First one", "Second one", "Third"], {"repeated": 1, "extra": 1},
    )
    assert parse_lines("", limit=3) == ([], {"repeated": 0, "extra": 0})


def test_parse_passage_joins_lines_into_one_text():
    assert parse_passage("Passage: Fog delayed the ferry.\n\nIt left at noon.") == (
        "Fog delayed the ferry. It left at noon."
    )


# --- The LLM transformations, with a scripted model ----------------------------------------


class Scripted:
    """Stands in for the LLM: replies with a fixed text and records what it was sent.
    Priced as a real model, so costs are not zero."""

    model = "gpt-6-luna"

    def __init__(self, reply: str = "", error: Exception | None = None):
        self.reply, self.error = reply, error
        self.requests: list[tuple[str, str]] = []

    def generate(self, system: str, user: str) -> Generation:
        self.requests.append((system, user))
        if self.error:
            raise self.error
        return Generation(self.reply, input_tokens=200, output_tokens=50, stop_reason="stop")


def test_rewriter_sends_the_question_alone_and_returns_one_query():
    llm = Scripted("Query: harbour ferry delay fog Tuesday\nAn extra line.")
    result = Rewriter(llm).transform("Why was the harbour ferry delayed on Tuesday?")
    assert result.queries == [Query("harbour ferry delay fog Tuesday", "rewrite")]
    assert result.notes == {"extra": 1, "stop_reason": "stop"}
    assert (result.calls, result.input_tokens, result.output_tokens) == (1, 200, 50)
    system, user = llm.requests[0]
    assert user == "Question: Why was the harbour ferry delayed on Tuesday?"
    assert "do not add any fact" in system


def test_multi_query_keeps_n_phrasings_and_searches_the_question_too_by_default():
    question = "What delayed the ferry?"
    llm = Scripted(f"1. Why was the ferry late?\n2. {question}\n3. What held the ferry up?")
    transformer = MultiQuery(llm, n=3)
    result = transformer.transform(question)
    # The phrasing that only repeats the question is dropped and counted as missing.
    assert result.queries == [
        Query("Why was the ferry late?", "paraphrase"),
        Query("What held the ferry up?", "paraphrase"),
    ]
    assert result.notes == {"missing": 1, "stop_reason": "stop"}
    assert transformer.retrieve_with == "both" and transformer.combine_method == "rrf"
    assert "exactly 3 lines" in llm.requests[0][0]
    assert [q.text for q in route(question, result.queries, "both")] == [
        question, "Why was the ferry late?", "What held the ferry up?",
    ]


def test_hyde_returns_the_passage_as_one_query():
    llm = Scripted("The harbour ferry was held up by fog.\nIt left at noon.")
    result = HyDE(llm).transform("What delayed the ferry?")
    assert result.queries == [
        Query("The harbour ferry was held up by fog. It left at noon.", "hypothetical")
    ]


def test_decomposer_caps_the_sub_questions_and_combines_by_round_robin_by_default():
    llm = Scripted("- A?\n- B?\n- C?")
    transformer = Decomposer(llm, max_queries=2)
    result = transformer.transform("A and B and C?")
    assert result.queries == [Query("A?", "sub-question"), Query("B?", "sub-question")]
    assert result.notes == {"extra": 1, "stop_reason": "stop"}
    assert transformer.combine_method == "round_robin"
    # A question the model returns unchanged is one fact, and is marked as the original.
    same = Decomposer(Scripted("Who won?")).transform("Who won?")
    assert same.queries == [Query("Who won?", "original")] and "fallback" not in same.notes


def test_a_refused_or_empty_reply_falls_back_to_the_question_and_other_errors_are_raised():
    class BadRequestError(Exception):
        pass

    refused = Rewriter(Scripted(error=BadRequestError("flagged"))).transform("Who won?")
    assert refused.queries == [Query("Who won?", "original")]
    assert refused.notes == {"fallback": True, "refused": True}
    assert (refused.calls, refused.input_tokens) == (0, 0)

    empty = Decomposer(Scripted("  \n ")).transform("Who won?")
    assert empty.queries == [Query("Who won?", "original")]
    assert empty.notes == {"fallback": True, "empty": True, "stop_reason": "stop"}
    assert empty.output_tokens == 50  # the call was still paid for

    # A missing key or a timeout is not swallowed.
    with pytest.raises(RuntimeError):
        Rewriter(Scripted(error=RuntimeError("no key"))).transform("Who won?")


def test_a_repeated_question_comes_from_the_cache_and_only_paid_calls_are_metered(tmp_path):
    llm = Scripted("ferry delay cause")
    transformer = Rewriter(llm, cache=tmp_path / "llm.sqlite")
    assert not transformer.is_cached("What delayed the ferry?")
    first = transformer.transform("What delayed the ferry?")
    assert transformer.is_cached("What delayed the ferry?")
    second = transformer.transform("What delayed the ferry?")
    assert (first.cached, second.cached) == (False, True)
    assert second.queries == first.queries and len(llm.requests) == 1
    assert transformer.meter.totals() == {"gpt-6-luna": (1, 200, 50)}
    assert transformer.meter.cost() == pytest.approx((200 * 0.10 + 50 * 0.50) / 1e6)


def test_meter_writes_what_it_counted_and_starts_again(tmp_path):
    meter = Meter()
    meter.add("gpt-6-luna", 1000, 100)
    meter.add("gpt-6-luna", 1000, 100)
    ledger = tmp_path / "ledger.csv"
    assert meter.flush(ledger, "step") == pytest.approx((2000 * 0.10 + 200 * 0.50) / 1e6)
    assert "step,gpt-6-luna,2000,200" in ledger.read_text()
    assert meter.totals() == {} and meter.flush(ledger, "step") == 0.0
    assert len(ledger.read_text().splitlines()) == 2  # the header and one row


# --- Which stage reads which query, and combining lists ------------------------------------


def test_route_gives_each_stage_the_question_the_queries_or_both():
    queries = [Query("a", "clause"), Query("b", "clause")]
    assert route("q", queries, "original") == [Query("q")]
    assert route("q", queries, "queries") == queries
    assert route("q", queries, "both") == [Query("q"), *queries]
    # The question is never listed twice, and a fallback leaves it as the only query.
    assert route("q", [Query("q")], "both") == [Query("q")]
    assert route("q", [], "queries") == [Query("q")]


def test_routing_refuses_settings_it_does_not_know():
    Routing(retrieve_with="both", rerank_with="queries", combine={"method": "rrf", "k": 10})
    for settings in (
        {"retrieve_with": "all"},
        {"rerank_with": "clauses"},
        {"combine": {"method": "borda"}},
        {"combine": {"method": "round_robin", "k": 60}},
        {"rerank_combine": "mean"},
    ):
        with pytest.raises(ValueError):
            Routing(**settings)
    with pytest.raises(ValueError, match="max_queries"):
        Decomposer(Scripted(), max_queries=1)


def ranked(*names: str) -> list[Candidate]:
    return [
        Candidate(Chunk(name, "doc", name, 0, 1), 1.0 / rank, rank, "hybrid")
        for rank, name in enumerate(names, start=1)
    ]


def test_round_robin_takes_each_lists_choices_in_turn_and_skips_repeats():
    lists = {"0": ranked("A", "B", "C"), "1": ranked("X", "A", "Y", "Z")}
    merged = round_robin(lists, "queries")
    # First choices A, X; second choices B (A is taken); third C, Y; then Z.
    assert ids(merged) == ["A", "X", "B", "C", "Y", "Z"]
    assert [c.rank for c in merged] == [1, 2, 3, 4, 5, 6]
    assert merged[0].score == 1.0 and merged[3].score == 0.25
    assert {c.retriever for c in merged} == {"queries"}
    assert round_robin({}) == []


def test_round_robin_keeps_a_chunk_only_one_list_wants_where_rank_fusion_would_not():
    from ragbasics.retrieval.fusion import reciprocal_rank_fusion

    # Three sub-questions. S, T, U are each the first choice of one; M is everyone's second.
    lists = {"0": ranked("S", "M"), "1": ranked("T", "M"), "2": ranked("U", "M")}
    assert ids(reciprocal_rank_fusion(lists, k=60))[0] == "M"  # 3/62 beats 1/61
    assert ids(round_robin(lists))[:3] == ["S", "T", "U"]


def test_merging_reranked_lists_by_best_score_and_by_round_robin():
    found = candidates("a", "b", "c", "d")
    # Sub-question 1 likes c then a; sub-question 2 scores everything higher and likes d.
    first = Reranked(reorder(found, [2.0, 0.0, 3.0, 0.0], "rerank"), {"rerank": 0.5},
                     notes={"truncated_pairs": 1})
    second = Reranked(reorder(found, [5.0, 6.0, 4.0, 9.0], "rerank"), {"rerank": 0.25},
                      cached=True, notes={"truncated_pairs": 2})
    best = merge(found, [first, second], "max")
    # The second sub-question's scores are all higher, so its order is the result.
    assert ids(best.candidates) == ["c4", "c2", "c1", "c3"]
    assert [c.score for c in best.candidates] == [9.0, 6.0, 5.0, 4.0]
    turns = merge(found, [first, second], "round_robin")
    assert ids(turns.candidates) == ["c3", "c4", "c1", "c2"]
    assert {c.retriever for c in turns.candidates} == {"rerank"}
    assert turns.timings == {"rerank": 0.75} and turns.cached is False
    assert turns.notes == {"truncated_pairs": 3, "queries": 2, "combine": "round_robin"}
    with pytest.raises(ValueError):
        merge(found, [first, second], "mean")


# --- In the pipeline -----------------------------------------------------------------------

QUESTION = "Did bakery revenue rise nine percent and what closed the airport?"


@register("generator", "split_on_and")
class SplitOnAnd:
    """Stands in for the LLM: replies with the question cut at " and ", one piece per
    line. Raises on a question that mentions trees, to test a run that dies half way."""

    model = "gpt-6-luna"
    calls = 0
    lock = threading.Lock()

    def generate(self, system: str, user: str) -> Generation:
        with self.lock:
            type(self).calls += 1
        question = user.removeprefix("Question: ")
        if "trees" in question:
            raise RuntimeError("the connection dropped")
        return Generation("\n".join(question.split(" and ")), 300, 40, "stop")


def transform(name: str, **params) -> dict:
    return {"name": name, "params": {"generator": {"name": "split_on_and"}, **params}}


def test_there_is_no_transformation_by_default_so_old_configs_are_unchanged(tmp_path):
    cfg = offline_config(tmp_path)
    assert cfg.query_transform is None and cfg.followup is None
    pipeline = Pipeline(cfg)
    pipeline.ingest(DOCS)
    trace = pipeline.ask("Why was the ferry delayed?")
    assert trace.queries == [] and trace.query_lists == [] and trace.rerank_queries == []
    assert trace.transform == {} and trace.asked == ""
    assert "transformer" not in trace.models
    assert set(trace.usage) == {"embed_tokens", "input_tokens", "output_tokens"}


def test_each_sub_question_is_searched_and_the_lists_are_combined(tmp_path):
    plain = Pipeline(offline_config(tmp_path, top_k=3))
    plain.ingest(DOCS)
    meta = (tmp_path / "index" / "meta.json").read_text()

    cfg = offline_config(tmp_path, top_k=3, query_transform=transform("decompose"))
    pipeline = Pipeline(cfg)
    assert pipeline.load() is True  # not part of what the index depends on
    assert (tmp_path / "index" / "meta.json").read_text() == meta
    trace = pipeline.ask(QUESTION)
    assert trace.queries == [
        Query("Did bakery revenue rise nine percent", "sub-question"),
        Query("what closed the airport?", "sub-question"),
    ]
    # Each sub-question's own list, and the two taken in turn.
    assert [found[0].chunk.doc_id for found in trace.query_lists] == ["doc_b", "doc_c"]
    assert [c.chunk.doc_id for c in trace.candidates[:2]] == ["doc_b", "doc_c"]
    assert [c.rank for c in trace.candidates] == [1, 2, 3]
    assert {c.retriever for c in trace.candidates} == {"queries"}
    assert trace.retrievers == {}  # one list per query, not per retriever
    assert trace.transform == {
        "name": "decompose", "retrieve_with": "queries", "rerank_with": "original",
        "calls": 1, "cached": False, "stop_reason": "stop", "combine": "round_robin",
    }
    assert trace.models["transformer"] == "gpt-6-luna"
    assert {"transform", "combine", "generate"} <= set(trace.timings)
    assert trace.usage["transform_input_tokens"] == 300
    assert trace.cost_usd == pytest.approx((300 * 0.10 + 40 * 0.50) / 1e6)
    # The generator answers the question as asked.
    assert trace.question == QUESTION
    assert trace.user_prompt.endswith(f"Question: {QUESTION}")


def test_both_searches_the_question_first_and_one_query_keeps_the_retrievers_lists(tmp_path):
    cfg = offline_config(
        tmp_path, top_k=3, retriever={"name": "hybrid"},
        query_transform=transform("decompose", retrieve_with="both", combine={"method": "rrf"}),
    )
    pipeline = Pipeline(cfg)
    pipeline.ingest(DOCS)
    trace = pipeline.retrieve(QUESTION)
    assert [q.role for q in trace.queries] == ["original", "sub-question", "sub-question"]
    assert len(trace.query_lists) == 3 and trace.transform["combine"] == "rrf"
    assert {c.retriever for found in trace.query_lists for c in found} == {"hybrid"}

    # A question with nothing to split gives one query, and the trace then holds the
    # dense and sparse lists as it does without a transformation.
    single = pipeline.retrieve("What closed the airport?")
    assert [q.role for q in single.queries] == ["original"]
    assert set(single.retrievers) == {"dense", "sparse"} and "combine" not in single.transform


def test_a_made_up_passage_is_embedded_as_a_document_and_a_question_as_a_query(tmp_path):
    cfg = offline_config(tmp_path, query_transform=transform("hyde", retrieve_with="both"))
    pipeline = Pipeline(cfg)
    pipeline.ingest(DOCS)
    kinds: list[tuple[str, str]] = []
    embed = pipeline.embedder.embed

    def spy(texts, kind="document"):
        kinds.extend((text, kind) for text in texts)
        return embed(texts, kind=kind)

    pipeline.embedder.embed = spy
    trace = pipeline.retrieve(QUESTION)
    assert [q.role for q in trace.queries] == ["original", "hypothetical"]
    passage = "Did bakery revenue rise nine percent what closed the airport?"
    assert kinds == [(QUESTION, "query"), (passage, "document")]


def test_the_reranker_scores_against_each_sub_question_when_the_config_says_so(tmp_path):
    cfg = offline_config(
        tmp_path, top_k=3,
        query_transform=transform("decompose", retrieve_with="original", rerank_with="queries"),
    )
    pipeline = Pipeline(cfg)
    pipeline.ingest(DOCS)
    scorer = WordOverlapScorer()
    pipeline.reranker = CrossEncoderReranker("stub", scorer=scorer, depth=3)
    trace = pipeline.retrieve(QUESTION)
    # The candidates are the plain retriever's: the question was the only search.
    assert trace.queries == [Query(QUESTION)] and len(trace.query_lists) == 1
    assert [q.role for q in trace.rerank_queries] == ["sub-question", "sub-question"]
    # One pass per sub-question over the same three candidates.
    asked = [sorted({q for q, _ in batch}) for batch in scorer.batches]
    assert asked == [["Did bakery revenue rise nine percent"], ["what closed the airport?"]]
    assert all(len(batch) == 3 for batch in scorer.batches)
    # Each sub-question's first choice, in turn.
    assert [c.chunk.doc_id for c in trace.candidates[:2]] == ["doc_b", "doc_c"]
    assert trace.rerank["queries"] == 2 and trace.rerank["combine"] == "round_robin"

    # With rerank_with left at "original" there is one pass, against the question.
    pipeline.transformer.rerank_with = "original"
    scorer.batches.clear()
    trace = pipeline.retrieve(QUESTION)
    assert [sorted({q for q, _ in batch}) for batch in scorer.batches] == [[QUESTION]]
    assert "queries" not in trace.rerank


def test_trace_formatter_shows_the_queries_and_each_chunks_rank_per_query(tmp_path):
    from ragbasics.cli import format_trace

    cfg = offline_config(tmp_path, top_k=2, query_transform=transform("decompose"))
    pipeline = Pipeline(cfg)
    pipeline.ingest(DOCS)
    text = format_trace(pipeline.ask(QUESTION))
    assert "Query transformation 'decompose' by gpt-6-luna:" in text
    assert "    1. [sub-question] Did bakery revenue rise nine percent" in text
    assert "(query 1: 1, query 2: 2)" in text
    assert "Reranked against" not in text  # no reranker in this config


# --- Chat follow-ups -----------------------------------------------------------------------


def test_followup_prompt_holds_the_last_turns_and_the_new_message():
    history = [("Who runs the ferry?", "Harbour Lines."), ("Is it late?", "Yes, by fog.")]
    assert build_prompt(history, "Since when?") == (
        "<conversation>\nUser: Who runs the ferry?\nAssistant: Harbour Lines.\n\n"
        "User: Is it late?\nAssistant: Yes, by fog.\n</conversation>\n\nNew message: Since when?"
    )
    assert "unchanged" in FOLLOWUP_SYSTEM


def test_followup_rewriter_calls_no_model_without_history_and_keeps_the_last_turns_only():
    llm = Scripted("Question: Since when has the harbour ferry been late?")
    rewriter = FollowUpRewriter(llm, turns=1)
    first = rewriter.rewrite([], "Is the ferry late?")
    assert (first.question, first.rewritten, llm.requests) == ("Is the ferry late?", False, [])

    history = [("Who runs the ferry?", "Harbour Lines."), ("Is it late?", "Yes, by fog.")]
    result = rewriter.rewrite(history, "Since when?")
    assert result.question == "Since when has the harbour ferry been late?"
    assert result.rewritten is True and result.input_tokens == 200
    assert "Who runs the ferry?" not in llm.requests[0][1]  # only the last turn is shown

    # A message returned unchanged is not a rewrite, and an empty reply keeps the message.
    same = FollowUpRewriter(Scripted("Since when?")).rewrite(history, "Since when?")
    assert same.rewritten is False
    empty = FollowUpRewriter(Scripted("")).rewrite(history, "Since when?")
    assert (empty.question, empty.notes) == ("Since when?", {"empty": True})


@register("generator", "names_the_airport")
class NamesTheAirport:
    model = "gpt-6-luna"

    def generate(self, system: str, user: str) -> Generation:
        return Generation("How long was the airport closed by volcanic ash?", 250, 15, "stop")


def test_a_follow_up_is_rewritten_before_retrieval_and_the_trace_keeps_both(tmp_path):
    cfg = offline_config(
        tmp_path, followup={"name": "llm", "params": {"generator": {"name": "names_the_airport"}}}
    )
    pipeline = Pipeline(cfg)
    pipeline.ingest(DOCS)
    # The first message of a chat is answered as it is.
    first = pipeline.ask("What closed the airport?", history=[])
    assert first.asked == "" and "followup" not in first.timings

    trace = pipeline.ask("For how long?", history=[(first.question, first.answer)])
    assert trace.asked == "For how long?"
    assert trace.question == "How long was the airport closed by volcanic ash?"
    assert trace.candidates[0].chunk.doc_id == "doc_c"
    assert trace.user_prompt.endswith(f"Question: {trace.question}")
    assert trace.models["followup"] == "gpt-6-luna" and "followup" in trace.timings
    assert trace.usage["followup_input_tokens"] == 250
    assert trace.cost_usd == pytest.approx((250 * 0.10 + 15 * 0.50) / 1e6)


# --- In the evaluator ----------------------------------------------------------------------


def with_transform(plain: PipelineConfig, tmp_path: Path, **params) -> PipelineConfig:
    spec = transform("decompose", cache=str(tmp_path / "cache.sqlite"), **params)
    return plain.model_copy(
        update={"name": "tiny-split", "query_transform": PipelineConfig.model_validate(
            plain.model_dump() | {"query_transform": spec}).query_transform}
    )


def test_a_transformed_run_records_the_queries_and_each_querys_list(evaluation):  # noqa: F811
    run, plain, _, _, tmp_path = evaluation
    SplitOnAnd.calls = 0
    answerable = with_transform(plain, tmp_path).model_copy(update={"name": "tiny-split"})
    # q3 mentions trees, which the stand-in model dies on; leave it for the next test.
    questions = [json.loads(line) for line in (tmp_path / "questions_dev.jsonl").open()]
    kept = [q for q in questions if "trees" not in q["text"]]
    (tmp_path / "questions_dev.jsonl").write_text("".join(json.dumps(q) + "\n" for q in kept))

    after = rows(run(answerable))
    row = after["q2"]
    assert row["queries"] == [
        ["sub-question", "Did bakery revenue rise"],
        ["sub-question", "did volcanic ash close the airport?"],
    ]
    assert row["rerank_queries"] == [["original", row["question"]]]
    assert row["transform"]["name"] == "decompose" and len(row["by_query"]) == 2
    assert "by_query" not in after["q1"]  # nothing to split: one query
    summary = json.loads((tmp_path / "runs" / "tiny-split" / "metrics.json").read_text())
    assert summary["models"]["transformer"] == "gpt-6-luna"
    # Four calls of 300 tokens in and 40 out, billed once and written to the ledger.
    assert summary["cost_usd"]["billed_this_run"] == pytest.approx(4 * 0.00005, abs=1e-6)
    ledger = (tmp_path / "ledger.csv").read_text()
    assert "eval:tiny-split,gpt-6-luna,1200,160" in ledger and "aborted" not in ledger
    # Every reply is now cached: a repeat calls nothing and bills nothing.
    run(answerable)
    assert SplitOnAnd.calls == 4 and (tmp_path / "ledger.csv").read_text() == ledger


def test_a_transformation_is_estimated_before_any_call(evaluation):  # noqa: F811
    run, plain, _, eval_cfg, tmp_path = evaluation
    SplitOnAnd.calls = 0
    cfg = with_transform(plain, tmp_path)
    with pytest.raises(EstimateOnly):
        run(cfg, estimate_only=True)
    with pytest.raises(ConfirmationNeeded):
        run(cfg, eval_cfg.model_copy(update={"confirm_above_usd": 0.0}))
    assert SplitOnAnd.calls == 0 and not (tmp_path / "runs").exists()


def test_a_run_that_dies_after_paid_calls_still_writes_them_to_the_ledger(evaluation):  # noqa: F811
    run, plain, _, _, tmp_path = evaluation
    SplitOnAnd.calls = 0
    with pytest.raises(RuntimeError, match="connection dropped"):
        run(with_transform(plain, tmp_path))
    assert not (tmp_path / "runs").exists()
    # Five questions, one of which failed: the four that were paid for are on record.
    assert SplitOnAnd.calls == 5
    ledger = (tmp_path / "ledger.csv").read_text()
    assert "eval:tiny-split (aborted run),gpt-6-luna,1200,160" in ledger


# --- Configs -------------------------------------------------------------------------------


def test_stage07_configs_change_only_the_query_transformation():
    plain = load_config(REPO_ROOT / "configs/stage05_retrieval/hybrid-rrf.yaml", PipelineConfig)
    reranked = {
        depth: load_config(REPO_ROOT / f"configs/stage06_rerank/qwen3-{depth}.yaml", PipelineConfig)
        for depth in (50, 100)
    }
    paths = sorted((REPO_ROOT / "configs" / "stage07_query").glob("*.yaml"))
    assert paths
    same = {"index_dir", "chunker", "embedder", "store", "retriever", "reranker", "top_k",
            "generator", "overfetch", "embedding_cache"}
    for path in paths:
        cfg = load_config(path, PipelineConfig)
        pipeline = Pipeline(cfg.model_copy(update={"embedding_cache": None}))  # loads no model
        assert cfg.name == f"query-{path.stem}"
        assert cfg.query_transform is not None
        assert cfg.query_transform.name.replace("_", "-") in path.stem
        base = reranked[cfg.reranker.params["depth"]] if cfg.reranker else plain
        assert ("rerank" in path.stem) == (cfg.reranker is not None)
        assert cfg.model_dump(include=same) == base.model_dump(include=same)
        # A row that changes what the reranker scores against keeps the candidates as
        # they were, and says so in its file name.
        transformer = pipeline.transformer
        split = transformer.rerank_with != "original"
        assert split == ("rerank-split" in path.stem)
        if split:
            assert transformer.retrieve_with == "original"
            assert path.stem.endswith(f"-{cfg.reranker.params['depth']}")
        assert ("-both" in path.stem or cfg.query_transform.name == "multi_query") == (
            transformer.retrieve_with == "both"
        )
