"""The evaluation end to end on a four-article corpus, with no API calls."""

import csv
import json
from pathlib import Path

import pytest

from ragbasics.config import EvalConfig, PipelineConfig, load_config
from ragbasics.eval import report
from ragbasics.eval.dataset import write_jsonl
from ragbasics.eval.runner import (
    ConfirmationNeeded,
    EstimateOnly,
    gold_context,
    run_eval,
    run_id,
)
from ragbasics.generation.llm import Generation
from ragbasics.pipeline import Pipeline
from ragbasics.registry import register
from ragbasics.types import Document, EvidenceSpan, Question

REPO_ROOT = Path(__file__).resolve().parents[1]

DOCS = [
    Document("doc_a", "The harbour ferry was delayed by fog on Tuesday."),
    Document("doc_b", "Quarterly revenue at the bakery chain rose nine percent."),
    Document("doc_c", "Volcanic ash closed the airport for three days."),
    Document("doc_d", "The city council voted to plant four hundred trees."),
]


def span(doc: Document, phrase: str) -> EvidenceSpan:
    start = doc.text.index(phrase)
    return EvidenceSpan(doc.doc_id, start, start + len(phrase))


QUESTIONS = [
    Question("q1", "What delayed the harbour ferry?", "Fog", "inference_query",
             (span(DOCS[0], "delayed by fog"),), "q1", True),
    Question("q2", "Did bakery revenue rise and did volcanic ash close the airport?", "Yes",
             "comparison_query", (span(DOCS[1], "rose nine percent"), span(DOCS[2], "closed")),
             "q2", True),
    Question("q3", "How many trees did the city council vote to plant?", "Four hundred",
             "inference_query", (span(DOCS[3], "four hundred trees"),), "q3", True),
    Question("q4", "Who won the chess final?", "Insufficient information.", "null_query",
             (), "q4", True),
    # Outside the answer subset: scored for retrieval only.
    Question("q5", "What closed the airport for three days?", "Volcanic ash",
             "inference_query", (span(DOCS[2], "Volcanic ash"),), "q2", False),
]


@register("generator", "scripted")
class ScriptedGenerator:
    """Stands in for the generator: right on q1, wordy on q2, declines everything else."""

    model = "scripted"

    def generate(self, system: str, user: str) -> Generation:
        question = user.rsplit("Question: ", 1)[1]
        if "ferry" in question:
            text = "Fog."
        elif "bakery" in question:
            text = "Both passages confirm it, so the answer is yes."
        else:
            text = "Insufficient information."
        return Generation(text, input_tokens=1000, output_tokens=10, stop_reason="end_turn")


class StubJudge:
    """Calls an answer correct if it contains the reference, and supported always."""

    model = "stub-judge"

    def __init__(self):
        self.calls = 0

    def generate(self, system: str, user: str) -> Generation:
        self.calls += 1
        if "Reference answer:" in user:
            reference = user.split("Reference answer: ")[1].split("\n")[0].lower().rstrip(".")
            candidate = user.split("Candidate answer: ")[1].lower()
            verdict = "correct" if reference in candidate else "incorrect"
        else:
            verdict = "supported"
        return Generation(json.dumps({"reasoning": "stub", "verdict": verdict}), 200, 20)


@pytest.fixture
def setup(tmp_path):
    write_jsonl(tmp_path / "documents.jsonl", DOCS)
    write_jsonl(tmp_path / "questions_dev.jsonl", QUESTIONS)
    pipeline_cfg = PipelineConfig.model_validate(
        {
            "name": "tiny",
            "index_dir": tmp_path / "index",
            "chunker": {"name": "fixed", "params": {"size": 64}},
            "embedder": {"name": "hashing", "params": {"dimensions": 256}},
            "store": {"name": "numpy"},
            "top_k": 2,
            "generator": {"name": "scripted"},
        }
    )
    eval_cfg = EvalConfig.model_validate(
        {
            "questions": tmp_path / "questions_dev.jsonl",
            "documents": tmp_path / "documents.jsonl",
            "depth": 4,
            "ks": [1, 2, 5],
            "rank_k": 10,
            "token_budget": 15,
            "judge": {"name": "openai", "params": {"model": "unused"}},
            "bootstrap": {"samples": 200, "seed": 0},
            "workers": 2,
            "cache": tmp_path / "cache.sqlite",
            "confirm_above_usd": 1.0,
        }
    )
    Pipeline(pipeline_cfg).ingest(DOCS)

    def run(**kwargs):
        kwargs.setdefault("judge", StubJudge())
        return run_eval(
            pipeline_cfg, eval_cfg, runs_dir=tmp_path / "runs", log=lambda _: None, **kwargs
        )

    return run, tmp_path, pipeline_cfg, eval_cfg


def read(run_dir: Path):
    summary = json.loads((run_dir / "metrics.json").read_text())
    with open(run_dir / "questions.jsonl") as f:
        rows = {row["question_id"]: row for row in map(json.loads, f)}
    return summary, rows


def test_repo_eval_config_loads():
    cfg = load_config(REPO_ROOT / "configs" / "eval.yaml", EvalConfig)
    assert cfg.questions.name == "questions_dev.jsonl"


def test_run_ids():
    assert run_id("baseline", "retrieved", 0) == "baseline"
    assert run_id("baseline", "retrieved", 1) == "baseline-r1"
    assert run_id("baseline", "gold", 0) == "baseline-gold"
    assert run_id("baseline", "guesser", 0) == "guesser"


def test_retrieved_run_scores_retrieval_on_all_and_answers_on_the_subset(setup):
    run, *_ = setup
    summary, rows = read(run())
    metrics = summary["metrics"]

    assert summary["run_id"] == "tiny"
    assert summary["questions"] == {
        "file": summary["questions"]["file"], "total": 5, "with_evidence": 4, "answered": 4,
    }
    # Retrieval: the four questions with evidence, including q5 outside the subset.
    assert metrics["recall@2"]["n"] == 4
    assert metrics["mrr@10"]["mean"] == 1.0  # each question's own article ranks first
    assert rows["q1"]["gold_ranks"] == [1]
    assert rows["q5"]["scores"]["recall@1"] == 1.0 and "answer" not in rows["q5"]
    assert "recall@1" not in rows["q4"]["scores"]  # no gold evidence, nothing to retrieve
    assert metrics["recall@15tok"]["n"] == 4 and 0 < metrics["precision@15tok"]["mean"] <= 1

    # Answers: q1 exact, q2 wordy and passed by the judge, q3 a false abstention, q4 a
    # correct abstention.
    assert rows["q1"]["match"] is True and rows["q2"]["match"] is None
    assert rows["q2"]["judge_correct"] is True
    assert [rows[q]["scores"]["correct"] for q in ("q1", "q2", "q3", "q4")] == [1, 1, 0, 1]
    assert metrics["correct"]["mean"] == 0.75 and metrics["correct"]["n"] == 4
    assert metrics["exact_match"]["mean"] == 0.5
    assert metrics["judged"]["mean"] == 0.25
    assert metrics["abstain_null"] == {"mean": 1.0, "low": 1.0, "high": 1.0, "n": 1}
    assert metrics["false_abstain"]["mean"] == pytest.approx(1 / 3)
    # q3's evidence was in the prompt, so this false abstention is the generator's.
    assert rows["q3"]["scores"]["false_abstain_with_support"] == 1.0
    assert metrics["judge_agrees_with_match"]["n"] == 3
    # Faithfulness is judged only where the answer asserts something.
    assert metrics["faithful"]["n"] == 2 and "judge_faithful" not in rows["q3"]
    assert summary["by_type"]["null_query"]["correct"]["n"] == 1
    assert summary["models"]["judge"] == "stub-judge"


def test_a_second_run_reads_everything_from_the_cache(setup):
    run, *_ = setup
    first, _ = read(run())
    judge = StubJudge()
    second, _ = read(run(judge=judge))
    assert judge.calls == 0
    assert second["metrics"] == first["metrics"]


def test_sample_one_is_a_fresh_draw_written_to_its_own_directory(setup):
    run, *_ = setup
    run()
    repeat = run(sample=1)
    assert repeat.name == "tiny-r1"
    assert report.flips(report.load_run(repeat), report.load_run(repeat.parent / "tiny")) == (0, 4)


def test_retrieval_only_makes_no_llm_calls(setup):
    run, *_ = setup
    judge = StubJudge()
    summary, rows = read(run(answers=False, judge=judge))
    assert judge.calls == 0 and "correct" not in summary["metrics"]
    assert summary["metrics"]["recall@2"]["n"] == 4


def test_gold_and_closed_book_runs(setup):
    run, tmp_path, _, _ = setup
    gold, gold_rows = read(run(mode="gold"))
    assert gold["run_id"] == "tiny-gold" and "recall@2" not in gold["metrics"]
    assert gold["metrics"]["correct"]["n"] == 4

    closed, closed_rows = read(run(mode="closed_book"))
    assert closed["run_id"] == "tiny-closed_book"
    assert "faithful" not in closed["metrics"]  # no passages to be faithful to


def test_gold_context_is_the_evidence_sentences():
    context = gold_context(QUESTIONS[1], {d.doc_id: d.text for d in DOCS})
    assert [c.text for c in context] == ["rose nine percent", "closed"]
    assert gold_context(QUESTIONS[3], {}) == []


def test_guesser_run_needs_no_models(setup):
    run, *_ = setup
    summary, rows = read(run(mode="guesser"))
    assert summary["run_id"] == "guesser" and summary["models"] == {}
    # Most common answer per type: inference has two different answers, so one is right.
    assert rows["q4"]["answer"] == "Insufficient information."
    assert summary["metrics"]["correct"]["n"] == 4


def test_run_above_the_cost_limit_needs_confirmation(setup):
    run, _, pipeline_cfg, eval_cfg = setup
    cheap_limit = eval_cfg.model_copy(update={"confirm_above_usd": -1.0})
    with pytest.raises(ConfirmationNeeded):
        run_eval(pipeline_cfg, cheap_limit, runs_dir=Path("unused"), judge=StubJudge(),
                 log=lambda _: None)


def test_estimate_only_stops_before_any_generator_or_judge_call(setup):
    run, tmp_path, _, _ = setup
    judge = StubJudge()
    with pytest.raises(EstimateOnly):
        run(estimate_only=True, judge=judge)
    assert judge.calls == 0
    assert not (tmp_path / "runs").exists()


def test_test_split_is_refused(setup):
    _, tmp_path, pipeline_cfg, eval_cfg = setup
    held_out = eval_cfg.model_copy(update={"questions": tmp_path / "questions_test.jsonl"})
    with pytest.raises(ValueError, match="held-out"):
        run_eval(pipeline_cfg, held_out, log=lambda _: None)


def test_report_tables_csv_and_readme(setup):
    run, tmp_path, _, _ = setup
    run()
    run(mode="gold")
    run(mode="guesser")
    runs = report.load_runs(tmp_path / "runs")
    table = report.results_markdown(runs)
    assert "**Retrieval** (4 development questions with gold evidence)" in table
    assert "| tiny-gold |" in table and "| guesser |" in table

    report.write_csv(runs, tmp_path / "results.csv")
    with open(tmp_path / "results.csv") as f:
        records = list(csv.DictReader(f))
    correct = [r for r in records if r["metric"] == "correct" and r["question_type"] == "all"]
    assert {r["run"] for r in correct} == {"tiny", "tiny-gold", "guesser"}

    readme = tmp_path / "README.md"
    readme.write_text(f"intro\n{report.START}\nSTALE\n{report.END}\noutro\n")
    assert report.update_readme(readme, table) is True
    text = readme.read_text()
    assert "STALE" not in text and text.startswith("intro\n") and text.endswith("outro\n")
    readme.write_text("no markers")
    assert report.update_readme(readme, table) is False


def test_compare_is_the_paired_difference_on_shared_questions(setup):
    run, tmp_path, _, _ = setup
    a, b = report.load_run(run()), report.load_run(run(mode="guesser"))
    diff = report.compare(a, b, "correct")
    assert diff.n == 4
    mine = sum(r["scores"]["correct"] for r in a["rows"] if "correct" in r["scores"])
    theirs = sum(r["scores"]["correct"] for r in b["rows"] if "correct" in r["scores"])
    assert diff.mean == pytest.approx((mine - theirs) / 4)
