"""Run one evaluation and write it to runs/<run id>/.

    retrieved    the pipeline as configured: retrieve, then answer from the top chunks
    closed_book  no retrieval; the generator answers from memory
    gold         the gold evidence sentences are given as the passages
    guesser      no model; the most common answer for the question's type

The last three are reference rows. Closed-book and gold bound what retrieval can add,
and the guesser is the floor that the answer distribution alone gives.

Retrieval metrics use every question that has gold evidence. Answer metrics use the
fixed answer subset, because each answer costs a generator call and judge calls.
"""

import json
import subprocess
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import tiktoken
import yaml

from ragbasics.cache import CachedGenerator, DiskCache
from ragbasics.config import EvalConfig, PipelineConfig
from ragbasics.context.prompts import (
    CLOSED_BOOK_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_closed_book_prompt,
    build_user_prompt,
)
from ragbasics.costs import (
    CLAUDE_TOKENS_PER_CL100K,
    append_ledger,
    embedding_cost,
    generation_cost,
)
from ragbasics.eval import retrieval_metrics as rm
from ragbasics.eval.answer_metrics import fit_guesser, is_abstention, match_verdict
from ragbasics.eval.dataset import read_documents, read_questions
from ragbasics.eval.judge import (
    CORRECTNESS_SYSTEM,
    FAITHFULNESS_SYSTEM,
    faithfulness_prompt,
    judge_correctness,
    judge_faithfulness,
)
from ragbasics.eval.stats import bootstrap_ci
from ragbasics.generation.llm import Generator
from ragbasics.pipeline import Pipeline
from ragbasics.registry import build
from ragbasics.types import Chunk, Question, Trace

MODES = ("retrieved", "closed_book", "gold", "guesser")
NULL_TYPE = "null_query"
# Used only for the estimate printed before a run.
ESTIMATED_ANSWER_TOKENS = 40
ESTIMATED_JUDGE_OUTPUT_TOKENS = 150
JUDGE_INSTRUCTION_TOKENS = 250


class ConfirmationNeeded(Exception):
    """The estimated cost is above the configured limit and the run was not confirmed."""


class EstimateOnly(Exception):
    """Raised after the estimate when the caller asked for the estimate and nothing else."""


def run_id(name: str, mode: str, sample: int) -> str:
    """'baseline', 'baseline-gold', 'baseline-r1' (a repeat of the same run)."""
    parts = [name] if mode != "guesser" else ["guesser"]
    if mode not in ("retrieved", "guesser"):
        parts.append(mode)
    if sample:
        parts.append(f"r{sample}")
    return "-".join(parts)


def gold_context(question: Question, text_by_doc: dict[str, str]) -> list[Chunk]:
    """The question's gold evidence sentences as passages, in their dataset order."""
    return [
        Chunk(
            chunk_id=f"gold:{span.doc_id}:{span.start_char}",
            doc_id=span.doc_id,
            text=text_by_doc[span.doc_id][span.start_char : span.end_char],
            start_char=span.start_char,
            end_char=span.end_char,
        )
        for span in question.evidence
    ]


def score_retrieval(
    question: Question,
    chunks: list[Chunk],
    total_relevant: int,
    cfg: EvalConfig,
    encoding: Any,
) -> dict[str, float]:
    """Every retrieval metric for one question that has gold evidence."""
    spans = question.evidence
    relevant = rm.relevance(chunks, spans)
    kept = rm.within_budget(chunks, cfg.token_budget, encoding)
    scores: dict[str, float] = {}
    for k in cfg.ks:
        scores[f"recall@{k}"] = rm.evidence_recall(chunks, spans, k)
        scores[f"full_support@{k}"] = rm.full_support(chunks, spans, k)
    scores[f"mrr@{cfg.rank_k}"] = rm.reciprocal_rank(relevant, cfg.rank_k)
    scores[f"ndcg@{cfg.rank_k}"] = rm.ndcg(relevant, total_relevant, cfg.rank_k)
    scores[f"recall@{cfg.token_budget}tok"] = rm.budget_recall(kept, spans)
    scores[f"precision@{cfg.token_budget}tok"] = rm.budget_precision(kept, spans)
    return scores


def count_relevant(index_chunks: list[Chunk], questions: list[Question]) -> dict[str, int]:
    """For each question, how many chunks of the whole index overlap its gold spans.
    nDCG needs it to know the best ranking that was possible. Chunks that return the
    same span (children of one parent) count once, as they do in a ranked list."""
    by_doc: dict[str, list[Chunk]] = {}
    seen: set[tuple[str, int, int]] = set()
    for chunk in index_chunks:
        span = (chunk.doc_id, chunk.start_char, chunk.end_char)
        if span not in seen:
            seen.add(span)
            by_doc.setdefault(chunk.doc_id, []).append(chunk)
    return {
        q.question_id: sum(
            any(rm.overlaps(chunk, span) for span in q.evidence)
            for doc_id in {span.doc_id for span in q.evidence}
            for chunk in by_doc.get(doc_id, [])
        )
        for q in questions
    }


def _aggregate(rows: list[dict[str, Any]], cfg: EvalConfig) -> dict[str, dict[str, Any]]:
    """Mean and cluster-bootstrap interval for every per-question score in `rows`."""
    names: list[str] = []
    for row in rows:
        names += [name for name in row["scores"] if name not in names]
    out = {}
    for name in names:
        scored = [row for row in rows if row["scores"].get(name) is not None]
        interval = bootstrap_ci(
            [float(row["scores"][name]) for row in scored],
            [row["cluster"] for row in scored],
            cfg.bootstrap.samples,
            cfg.bootstrap.seed,
        )
        out[name] = interval.as_dict()
    return out


def _git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False
        )
        return result.stdout.strip()
    except OSError:
        return ""


def _parallel(fn: Callable[[Any], Any], items: list[Any], workers: int) -> list[Any]:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, items))


def run_eval(
    pipeline_cfg: PipelineConfig,
    eval_cfg: EvalConfig,
    mode: str = "retrieved",
    sample: int = 0,
    answers: bool = True,
    confirmed: bool = False,
    estimate_only: bool = False,
    allow_test: bool = False,
    runs_dir: Path = Path("runs"),
    ledger: Path | None = None,
    judge: Generator | None = None,
    log: Callable[[str], None] = print,
) -> Path:
    """Evaluate one pipeline in one mode. Returns the run directory.

    `answers=False` scores retrieval only, which costs almost nothing. `sample` selects
    an independent repeat of the generator calls. `judge` replaces the configured judge
    (tests pass a stub). `estimate_only=True` stops after printing the cost estimate,
    before any generator or judge call.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode '{mode}'; choose from {MODES}")
    if eval_cfg.questions.name.startswith("questions_test") and not allow_test:
        raise ValueError(
            f"{eval_cfg.questions} is held-out test data. It is used once, in Stage 12."
        )

    previous = runs_dir / run_id(pipeline_cfg.name, mode, sample) / "metrics.json"
    has_answers = previous.exists() and "correct" in json.loads(previous.read_text())["metrics"]
    if not answers and has_answers:
        raise ValueError(
            f"{previous.parent} holds an answer-quality run. A retrieval-only run has the "
            "same run id and would overwrite it."
        )

    questions = read_questions(eval_cfg.questions)
    subset = [q for q in questions if q.answer_subset]
    encoding = tiktoken.get_encoding("cl100k_base")
    rows: dict[str, dict[str, Any]] = {
        q.question_id: {
            "question_id": q.question_id,
            "question_type": q.question_type,
            "cluster": q.cluster,
            "answer_subset": q.answer_subset,
            "question": q.text,
            "reference": q.answer,
            "scores": {},
        }
        for q in questions
    }
    models: dict[str, str] = {}
    spend: dict[str, list[float]] = {}  # model -> [input tokens, output tokens, dollars] billed

    def bill(model: str, cached: bool, tokens_in: int, tokens_out: int) -> None:
        if cached:
            return
        totals = spend.setdefault(model, [0, 0, 0.0])
        totals[0] += tokens_in
        totals[1] += tokens_out
        totals[2] += generation_cost(model, tokens_in, tokens_out)

    # --- The guesser needs no pipeline ----------------------------------------------------
    if mode == "guesser":
        guesses = fit_guesser(questions)
        # Fitted on every question, scored on the answer subset like every other row.
        answered = subset
        for q in answered:
            row = rows[q.question_id]
            row["answer"] = guesses[q.question_type]
            row["match"] = match_verdict(row["answer"], q.answer)
            row["scores"]["correct"] = float(row["match"] is True)
        traces: dict[str, Trace] = {}
    else:
        pipeline = Pipeline(pipeline_cfg)
        models = {"embedder": pipeline.embedder.model, "generator": pipeline.generator.model}
        traces = {}

        # --- Retrieval, for every question -------------------------------------------------
        if mode == "retrieved":
            if not pipeline.load():
                raise ValueError(f"No index in {pipeline.index_dir}. Run `rag ingest` first.")
            log(f"Retrieving {eval_cfg.depth} chunks for {len(questions)} questions…")
            retrieved = _parallel(
                lambda q: pipeline.retrieve(q.text, eval_cfg.depth), questions, eval_cfg.workers
            )
            traces = {q.question_id: t for q, t in zip(questions, retrieved, strict=True)}
            total_relevant = count_relevant(pipeline.store.chunks, questions)
            embed_tokens = sum(t.usage["embed_tokens"] for t in retrieved)
            embed_cost = embedding_cost(pipeline.embedder.model, embed_tokens)
            if embed_cost:
                spend[pipeline.embedder.model] = [embed_tokens, 0, embed_cost]
            for q in questions:
                trace, row = traces[q.question_id], rows[q.question_id]
                chunks = [c.chunk for c in trace.candidates]
                row["retrieved"] = [[c.chunk.chunk_id, round(c.score, 4)] for c in trace.candidates]
                if q.evidence:
                    row["gold_ranks"] = rm.gold_ranks(chunks, q.evidence)
                    row["scores"] |= score_retrieval(
                        q, chunks, total_relevant[q.question_id], eval_cfg, encoding
                    )
        answered = subset if answers else []

    # --- Answers, for the answer subset ----------------------------------------------------
    if mode != "guesser" and answered:
        cache = DiskCache(eval_cfg.cache)
        generator = CachedGenerator(pipeline.generator, cache, sample)
        pipeline.generator = generator
        judge_llm = CachedGenerator(judge or build("generator", **_spec(eval_cfg)), cache)
        models["judge"] = judge_llm.model

        text_by_doc: dict[str, str] = {}
        if mode == "gold":
            text_by_doc = {d.doc_id: d.text for d in read_documents(eval_cfg.documents)}

        def context_for(q: Question) -> list[Chunk] | None:
            if mode == "closed_book":
                return None
            if mode == "gold":
                return gold_context(q, text_by_doc)
            top = traces[q.question_id].candidates[: pipeline_cfg.top_k]
            return [c.chunk for c in top]

        contexts = {q.question_id: context_for(q) for q in answered}

        # Estimate before spending. Generator prompts are known exactly; the judge's
        # depend on the answers, so they are estimated as if none were cached.
        def tokens(text: str) -> int:
            return len(encoding.encode(text, disallowed_special=()))

        gen_in = gen_calls = judge_in = 0
        for q in answered:
            context = contexts[q.question_id]
            if context is None:
                system, user = CLOSED_BOOK_SYSTEM_PROMPT, build_closed_book_prompt(q.text)
            else:
                system, user = SYSTEM_PROMPT, build_user_prompt(q.text, context)
            if not generator.is_cached(system, user):
                gen_calls += 1
                gen_in += tokens(system) + tokens(user)
            judge_in += JUDGE_INSTRUCTION_TOKENS + tokens(q.text) + ESTIMATED_ANSWER_TOKENS
            if context:
                judge_in += JUDGE_INSTRUCTION_TOKENS
                judge_in += tokens(faithfulness_prompt(q.text, "", context))
        gen_estimate = generation_cost(
            generator.model,
            int(gen_in * CLAUDE_TOKENS_PER_CL100K),
            gen_calls * ESTIMATED_ANSWER_TOKENS,
        )
        judge_calls = len(answered) + sum(bool(c) for c in contexts.values())
        judge_estimate = generation_cost(
            judge_llm.model, judge_in, judge_calls * ESTIMATED_JUDGE_OUTPUT_TOKENS
        )
        estimate = gen_estimate + judge_estimate
        log(
            f"Estimated cost: ${estimate:.2f}. Generator {generator.model}: {gen_calls} new "
            f"calls of {len(answered)} (${gen_estimate:.2f}). Judge {judge_llm.model}: up to "
            f"{judge_calls} calls (${judge_estimate:.2f}, less whatever is cached)."
        )
        if estimate_only:
            raise EstimateOnly(f"Estimate only: nothing was generated or judged ({mode}).")
        if estimate > eval_cfg.confirm_above_usd and not confirmed:
            raise ConfirmationNeeded(
                f"The estimate is above ${eval_cfg.confirm_above_usd:.2f}. "
                "Run again with --yes to go ahead."
            )

        def answer(q: Question) -> Trace:
            trace = traces.get(q.question_id) or Trace(question=q.text)
            return pipeline.generate(trace, contexts[q.question_id])

        log(f"Generating {len(answered)} answers…")
        for q, trace in zip(answered, _parallel(answer, answered, eval_cfg.workers), strict=True):
            traces[q.question_id] = trace
            bill(
                generator.model,
                trace.cached,
                trace.usage["input_tokens"],
                trace.usage["output_tokens"],
            )
            row = rows[q.question_id]
            row["answer"] = trace.answer
            row["stop_reason"] = trace.stop_reason
            row["usage"] = trace.usage
            row["cost_usd"] = round(trace.cost_usd, 6)
            row["match"] = match_verdict(trace.answer, q.answer)

        def grade(q: Question) -> tuple[Any, Any]:
            trace = traces[q.question_id]
            correct = judge_correctness(judge_llm, q.text, q.answer, trace.answer)
            # An answer that declines asserts nothing, so there is nothing to support.
            faithful = None
            if trace.context and not is_abstention(trace.answer):
                faithful = judge_faithfulness(judge_llm, q.text, trace.answer, trace.context)
            return correct, faithful

        def bill_judge(g: Any) -> None:
            bill(judge_llm.model, g.cached, g.input_tokens, g.output_tokens)

        log("Judging…")
        graded = _parallel(grade, answered, eval_cfg.workers)
        for q, (correct, faithful) in zip(answered, graded, strict=True):
            row = rows[q.question_id]
            bill_judge(correct.generation)
            row["judge_correct"] = correct.passed
            row["judge_correct_reason"] = correct.reasoning
            # The string comparison decides when it can; the judge decides the rest.
            verdict = row["match"] if row["match"] is not None else correct.passed
            row["scores"]["correct"] = float(verdict is True)
            if row["match"] is not None and correct.passed is not None:
                # A free check on the judge: it should agree wherever the answer is certain.
                row["scores"]["judge_agrees_with_match"] = float(correct.passed == row["match"])
            if faithful is not None:
                bill_judge(faithful.generation)
                row["judge_faithful"] = faithful.passed
                row["judge_faithful_reason"] = faithful.reasoning
                if faithful.passed is not None:
                    row["scores"]["faithful"] = float(faithful.passed)

    # --- Scores derived from the answer ------------------------------------------------------
    top_k_support = f"full_support@{pipeline_cfg.top_k}"
    for q in answered:
        row = rows[q.question_id]
        scores = row["scores"]
        abstained = is_abstention(row["answer"])
        row["abstained"] = abstained
        scores["exact_match"] = float(row["match"] is True)
        scores["judged"] = float(row["match"] is None)
        if q.question_type == NULL_TYPE:
            scores["abstain_null"] = scores["correct"]
        else:
            scores["correct_answerable"] = scores["correct"]
            scores["false_abstain"] = float(abstained)
            if top_k_support in scores:
                # Splits false abstentions by cause: the evidence was in the prompt and
                # the generator declined anyway, or retrieval never delivered it.
                supported = scores[top_k_support] == 1.0
                scores["false_abstain_with_support"] = float(abstained and supported)
                scores["false_abstain_without_support"] = float(abstained and not supported)

    # --- Aggregate and write -----------------------------------------------------------------
    all_rows = list(rows.values())
    types = sorted({q.question_type for q in questions})
    spent = sum(totals[2] for totals in spend.values())
    summary = {
        "run_id": run_id(pipeline_cfg.name, mode, sample),
        "name": pipeline_cfg.name,
        "mode": mode,
        "sample": sample,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "models": models,
        "questions": {
            "file": str(eval_cfg.questions),
            "total": len(questions),
            "with_evidence": sum(bool(q.evidence) for q in questions),
            "answered": len(answered),
        },
        "cost_usd": {
            "list_price": round(sum(row.get("cost_usd", 0.0) for row in all_rows), 4),
            "billed_this_run": round(spent, 4),
        },
        "metrics": _aggregate(all_rows, eval_cfg),
        "by_type": {
            t: _aggregate([r for r in all_rows if r["question_type"] == t], eval_cfg)
            for t in types
        },
    }

    out = runs_dir / summary["run_id"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "pipeline": pipeline_cfg.model_dump(mode="json"),
                "eval": eval_cfg.model_dump(mode="json"),
                "judge_prompts": {
                    "correctness": CORRECTNESS_SYSTEM,
                    "faithfulness": FAITHFULNESS_SYSTEM,
                },
            },
            sort_keys=False,
        )
    )
    with open(out / "questions.jsonl", "w") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    if ledger:
        for model, (tokens_in, tokens_out, dollars) in spend.items():
            if dollars:
                step = f"eval:{summary['run_id']}"
                append_ledger(ledger, step, model, int(tokens_in), int(tokens_out), dollars)
    log(f"Billed this run: ${spent:.4f}. Wrote {out}/")
    return out


def _spec(eval_cfg: EvalConfig) -> dict[str, Any]:
    return {"name": eval_cfg.judge.name, **eval_cfg.judge.params}
