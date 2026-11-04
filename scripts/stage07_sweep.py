"""Stage 7 tables: query transformations with and without the reranker, from runs/.

    python scripts/stage07_sweep.py            # print the tables
    python scripts/stage07_sweep.py --write    # also write results/stage07_query.csv
    python scripts/stage07_sweep.py --timing   # also time each transformation (a few cents)

Every run whose id starts with `query-` is a row. What it is compared with follows from
its own config, kept in runs/<id>/config.yaml:

    no reranker                              retr-hybrid-rrf: hybrid retrieval, the question
    reranker, scoring against the question   rerank-qwen3-<depth>: the same reranker and depth
    reranker, scoring against the parts      rerank-qwen3-<depth>, the same candidates

Every row ranks the same 15,149 chunks, so recall@k, MRR and nDCG@10 are fair between rows
as well as the token-budget metrics. The forms of each transformation were chosen on the
pool questions by scripts/stage07_tune.py; the rows here are the development questions,
reported once.
"""

import argparse
import csv
import statistics
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv

from ragbasics.config import PipelineConfig, load_config
from ragbasics.costs import Meter, append_ledger, embedding_cost, generation_cost
from ragbasics.eval import report
from ragbasics.eval.dataset import read_questions
from ragbasics.pipeline import Pipeline
from ragbasics.query.base import route
from ragbasics.registry import build
from ragbasics.types import Query
from stage05_sweep import BASELINE, DEEP, HYBRID, NDCG, RECALL, TYPES, _diff, _pct, diff, scored

RUNS = Path("runs")
CONFIGS = Path("configs/stage07_query")
QUESTIONS = Path("data/processed/questions_dev.jsonl")
CSV = Path("results/stage07_query.csv")
LEDGER = Path("runs/cost_ledger.csv")
FULL = "full_support@10"
TIMED_QUESTIONS = 20  # LLM calls per transformation, with the cache off
TIMED_RERANK = 8  # questions reranked against their parts, with the score cache off
FIELDS = [
    "run", "family", "transform", "retrieve_with", "rerank_with", "combine", "depth",
    "against", "recall_2000tok", "recall_vs_ref", "recall_vs_ref_low", "recall_vs_ref_high",
    "recall_vs_baseline", "full_support_2000tok", "ndcg@10", "ndcg_vs_ref", "ndcg_vs_ref_low",
    "ndcg_vs_ref_high", FULL, "full_vs_ref", "full_vs_ref_low", "full_vs_ref_high", "mrr@10",
    "recall@10", DEEP, *[f"{label}_{suffix}" for label in TYPES.values()
                         for suffix in ("recall", "vs_ref", "vs_ref_low", "vs_ref_high")],
    "queries_per_question", "llm_calls_per_question", "fallbacks", "refused", "billed_usd",
]


def describe(run_id: str) -> dict:
    """What a run did, read from the config it was run with."""
    pipeline = yaml.safe_load((RUNS / run_id / "config.yaml").read_text())["pipeline"]
    spec = pipeline.get("query_transform") or {}
    params = spec.get("params", {})
    reranker = pipeline.get("reranker")
    depth = reranker["params"]["depth"] if reranker else ""
    rerank_with = params.get("rerank_with", "original")
    if not reranker:
        family, against = "no reranker", HYBRID
    elif rerank_with == "original":
        family, against = "in front of the reranker", f"rerank-qwen3-{depth}"
    else:
        family, against = "reranker scores the parts", f"rerank-qwen3-{depth}"
    combine = (params.get("rerank_combine", "round_robin") if family.startswith("reranker")
               else (params.get("combine") or {}).get("method", ""))
    return {"transform": spec.get("name", ""), "family": family, "against": against,
            "retrieve_with": params.get("retrieve_with", "queries"), "rerank_with": rerank_with,
            "combine": combine, "depth": depth}


def table(runs: dict[str, dict]) -> list[dict]:
    rows = []
    for run_id in sorted(r for r in runs if r.startswith("query-")):
        run, what = runs[run_id], describe(run_id)
        reference = runs.get(what["against"])
        if reference is None:
            print(f"{run_id}: no run {what['against']} to compare with; skipped.")
            continue
        by_question = scored(run)
        metrics = run["metrics"]
        notes = [r.get("transform", {}) for r in run["rows"]]
        row = {
            "run": run_id, **what,
            "recall_2000tok": metrics[RECALL]["mean"],
            **diff(run, reference, RECALL, "recall_vs_ref"),
            **diff(run, runs[BASELINE], RECALL, "recall_vs_baseline"),
            "full_support_2000tok": statistics.fmean(
                r["scores"][RECALL] == 1.0 for r in by_question.values()
            ),
            "ndcg@10": metrics[NDCG]["mean"],
            **diff(run, reference, NDCG, "ndcg_vs_ref"),
            FULL: metrics[FULL]["mean"],
            **diff(run, reference, FULL, "full_vs_ref"),
            "mrr@10": metrics["mrr@10"]["mean"],
            "recall@10": metrics["recall@10"]["mean"],
            DEEP: statistics.fmean(r["scores"][DEEP] for r in by_question.values()),
            "queries_per_question": statistics.fmean(
                len(r.get("rerank_queries" if what["family"].startswith("reranker") else
                          "queries") or [1]) for r in run["rows"]
            ),
            "llm_calls_per_question": statistics.fmean(n.get("calls", 0) for n in notes),
            "fallbacks": sum(bool(n.get("fallback")) for n in notes),
            "refused": sum(bool(n.get("refused")) for n in notes),
            "billed_usd": run["cost_usd"]["billed_this_run"],
        }
        for kind, label in TYPES.items():
            row[f"{label}_recall"] = run["by_type"][kind][RECALL]["mean"]
            row |= diff(run, reference, RECALL, f"{label}_vs_ref", kind)
        rows.append(row)
    return rows


def markdown(rows: list[dict], runs: dict[str, dict]) -> str:
    lines = []
    for family in ("no reranker", "in front of the reranker", "reranker scores the parts"):
        chosen = [r for r in rows if r["family"] == family]
        if not chosen:
            continue
        lines += [
            f"**{family}** (against {', '.join(sorted({r['against'] for r in chosen}))})", "",
            "| run | searched with | combined | recall in 2,000 tok | vs reference | vs baseline "
            "| full support in 2,000 tok | nDCG@10 | vs reference | full support@10 "
            "| vs reference | recall@50 | queries | fallbacks |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for r in chosen:
            searched = r["retrieve_with"] if family != "reranker scores the parts" else (
                f"reranker: {r['rerank_with']}")
            lines.append(
                f"| {r['run']} | {searched} | {r['combine']} | {_pct(r['recall_2000tok'])} "
                f"| {_diff(r, 'recall_vs_ref')} | {_diff(r, 'recall_vs_baseline')} "
                f"| {_pct(r['full_support_2000tok'])} | {_pct(r['ndcg@10'])} "
                f"| {_diff(r, 'ndcg_vs_ref')} | {_pct(r[FULL])} | {_diff(r, 'full_vs_ref')} "
                f"| {_pct(r[DEEP])} | {r['queries_per_question']:.2f} | {r['fallbacks']} |"
            )
        lines += ["", "Recall in 2,000 tokens by question type, with the paired difference "
                  "from the reference:", "", "| run | " + " | ".join(TYPES.values()) + " |",
                  "|---|" + "---|" * len(TYPES)]
        for r in chosen:
            cells = [f"{_pct(r[f'{label}_recall'])}, {_diff(r, f'{label}_vs_ref')}"
                     for label in TYPES.values()]
            lines.append(f"| {r['run']} | " + " | ".join(cells) + " |")
        lines.append("")
    # Does the gain survive the reranker? Each transformation's difference from its
    # reference, with and without the reranker behind it.
    pairs = []
    for r in rows:
        if r["family"] == "in front of the reranker":
            plain = r["run"].removesuffix("-rerank")
            match = next((x for x in rows if x["run"] == plain), None)
            if match:
                pairs.append((match, r))
    if pairs:
        lines += ["**Does the gain survive the reranker?** Recall in 2,000 tokens against "
                  "each row's own reference.", "",
                  "| transformation | no reranker | in front of the reranker "
                  "| nDCG@10, no reranker | nDCG@10, in front of the reranker |",
                  "|---|---|---|---|---|"]
        for plain, reranked in pairs:
            lines.append(f"| {plain['run'].removeprefix('query-')} "
                         f"| {_diff(plain, 'recall_vs_ref')} | {_diff(reranked, 'recall_vs_ref')} "
                         f"| {_diff(plain, 'ndcg_vs_ref')} | {_diff(reranked, 'ndcg_vs_ref')} |")
    return "\n".join(lines)


def timing() -> str:
    """Added latency per question, with every cache off: the LLM call, the extra
    retrievals, and the extra reranking passes. Costs a few cents; written to the ledger."""
    questions = [q for q in read_questions(QUESTIONS) if q.evidence][:TIMED_QUESTIONS]
    plain = Pipeline(load_config("configs/stage05_retrieval/hybrid-rrf.yaml", PipelineConfig)
                     .model_copy(update={"embedding_cache": None}))
    plain.load()
    lines = ["| transformation | LLM calls | LLM median s | LLM 95th percentile s | tokens in "
             "| tokens out | dollars per question | queries searched | retrieval median ms "
             "| question alone ms |", "|---|---|---|---|---|---|---|---|---|---|"]
    meters: list[Meter] = []
    embed_tokens = 0
    try:
        alone = []
        for q in questions:
            started = time.perf_counter()
            embed_tokens += plain.first_stage(q.text, 50).usage["embed_tokens"]
            alone.append(1000 * (time.perf_counter() - started))
        for key in ("rewrite", "multi-query", "hyde", "decompose", "clauses"):
            spec = load_config(CONFIGS / f"{key}.yaml", PipelineConfig).query_transform
            params = {k: v for k, v in spec.params.items() if k != "cache"}  # no LLM cache
            transformer = build("query_transform", spec.name, **params)
            if hasattr(transformer, "meter"):
                meters.append(transformer.meter)
            calls, tokens_in, tokens_out, searched, retrieval = [], [], [], [], []
            for q in questions:
                started = time.perf_counter()
                result = transformer.transform(q.text)
                calls.append(time.perf_counter() - started)
                tokens_in.append(result.input_tokens)
                tokens_out.append(result.output_tokens)
                queries = route(q.text, result.queries, transformer.retrieve_with)
                searched.append(len(queries))
                started = time.perf_counter()
                for query in queries:
                    kind = "document" if query.role == "hypothetical" else "query"
                    embed_tokens += plain.retriever.retrieve(
                        query.text, 100 if len(queries) > 1 else 50, plain, kind
                    ).embed_tokens
                retrieval.append(1000 * (time.perf_counter() - started))
            calls.sort()
            dollars = generation_cost(transformer.model, statistics.fmean(tokens_in),
                                      statistics.fmean(tokens_out))
            llm = transformer.model != "rule"
            lines.append(
                f"| {key} | {1 if llm else 0} | {statistics.median(calls):.2f} "
                f"| {calls[int(0.95 * (len(calls) - 1))]:.2f} "
                f"| {statistics.fmean(tokens_in):.0f} | {statistics.fmean(tokens_out):.0f} "
                f"| ${dollars:.5f} | {statistics.fmean(searched):.2f} "
                f"| {statistics.median(retrieval):.0f} | {statistics.median(alone):.0f} |"
            )
        lines += ["", reranking_time(questions[:TIMED_RERANK])]
    finally:
        for meter in meters:
            meter.flush(LEDGER, "stage07:timing")
        cost = embedding_cost(plain.embedder.model, embed_tokens)
        if cost:
            append_ledger(LEDGER, "stage07:timing", plain.embedder.model, embed_tokens, 0, cost)
    return "\n".join(lines)


def reranking_time(questions: list) -> str:
    """Qwen3 at depth 50 with the score cache off: one pass against the question, then
    the split configs' passes (the question and each part). Uses cached transformations;
    computes every score again."""
    spec = load_config("configs/stage06_rerank/qwen3-50.yaml", PipelineConfig).reranker
    reranker = build("reranker", spec.name, **{k: v for k, v in spec.params.items()
                                                 if k != "cache"})
    plain = Pipeline(load_config("configs/stage05_retrieval/hybrid-rrf.yaml", PipelineConfig))
    plain.load()
    first = [plain.first_stage(q.text, 50).candidates for q in questions]
    reranker.rerank(questions[0].text, first[0][:20])  # load the weights, warm the GPU
    lines = ["| reranker scores against | passes per question | median s per question |",
             "|---|---|---|"]
    for key in ("question", "clauses", "decompose"):
        times, passes = [], []
        transformer = None
        if key != "question":
            spec = load_config(CONFIGS / f"{key}-rerank-split-50.yaml",
                               PipelineConfig).query_transform
            transformer = build("query_transform", spec.name, **spec.params)
        for q, found in zip(questions, first, strict=True):
            queries = ([Query(q.text)] if transformer is None else route(
                q.text, transformer.transform(q.text).queries, transformer.rerank_with))
            started = time.perf_counter()
            for query in queries:
                reranker.rerank(query.text, found[:50])
            times.append(time.perf_counter() - started)
            passes.append(len(queries))
        lines.append(f"| {key} | {statistics.fmean(passes):.2f} | {statistics.median(times):.1f} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--timing", action="store_true")
    args = parser.parse_args()
    load_dotenv()
    runs = {r["run_id"]: r for r in report.load_runs(RUNS)}
    rows = table(runs)
    print(markdown(rows, runs))
    if args.timing:
        print(f"\nTiming {TIMED_QUESTIONS} questions with the caches off: about "
              f"{4 * TIMED_QUESTIONS} LLM calls to gpt-6-luna, about $0.02.")
        print("\n" + timing())
    if args.write:
        with open(CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore", restval="")
            writer.writeheader()
            for row in rows:
                writer.writerow({k: round(v, 4) if isinstance(v, float) else v
                                 for k, v in row.items()})
        print(f"\nWrote {CSV}")


if __name__ == "__main__":
    main()
