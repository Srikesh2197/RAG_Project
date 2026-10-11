"""Stage 6 tables: rerankers and candidate depths behind hybrid retrieval, from runs/.

    python scripts/stage06_sweep.py            # print the tables
    python scripts/stage06_sweep.py --write    # also write results/stage06_rerank.csv
    python scripts/stage06_sweep.py --timing   # also time each local reranker

Every run reranks the list of `retr-hybrid-rrf` over the 15,149 chunks of
data/index/chunk-recursive-128, so recall@k, MRR and nDCG@10 are comparable between
rows as well as the token-budget metrics. Depths 20 and 50 reorder chunks that were
already among the 50 the evaluator scores, so recall@50 cannot change; depth 100 can
bring in chunks the retriever ranked 51 to 100.

The model and depth were chosen on the pool questions by scripts/stage06_tune.py. The
rows here are the development questions, reported once.
"""

import argparse
import csv
import statistics
import time
from pathlib import Path

from ragbasics.config import PipelineConfig, load_config
from ragbasics.eval import report
from ragbasics.eval.dataset import read_questions
from ragbasics.pipeline import Pipeline
from ragbasics.registry import build
from stage05_sweep import (
    BASELINE,
    DEEP,
    HYBRID,
    NDCG,
    PRECISION,
    RECALL,
    TYPES,
    _diff,
    _pct,
    diff,
    scored,
)

RUNS = Path("runs")
CONFIGS = Path("configs/stage06_rerank")
QUESTIONS = Path("data/processed/questions_dev.jsonl")
WINS = Path("results/stage05_wins.csv")
CSV = Path("results/stage06_rerank.csv")
FULL = "full_support@10"

ORDER = [
    HYBRID,
    *[f"rerank-{model}-{depth}" for model in ("minilm", "qwen3", "llm-luna")
      for depth in (20, 50, 100)],
    "rerank-qwen3-100-weighted", "rerank-qwen3-100-header", "rerank-qwen3-100-qwen3-embed",
]
# Extra comparisons: (run, against, what the difference isolates).
PAIRS = [
    ("rerank-qwen3-100", "rerank-qwen3-50", "Qwen3: depth 100 instead of 50"),
    ("rerank-qwen3-50", "rerank-qwen3-20", "Qwen3: depth 50 instead of 20"),
    ("rerank-llm-luna-50", "rerank-qwen3-50", "the LLM instead of Qwen3, depth 50"),
    ("rerank-llm-luna-20", "rerank-qwen3-20", "the LLM instead of Qwen3, depth 20"),
    ("rerank-qwen3-100-weighted", "rerank-qwen3-100", "weighted fusion in front of Qwen3"),
    ("rerank-qwen3-100-header", "rerank-qwen3-100", "Qwen3 also reads title, source and date"),
    ("rerank-qwen3-100-qwen3-embed", "rerank-qwen3-100", "Qwen3 embeddings in the retriever"),
]
TIMED = {"minilm": {20: 100, 50: 100, 100: 100}, "qwen3": {20: 20, 50: 20, 100: 8}}
FIELDS = [
    "run", "reranker", "depth",
    "recall_2000tok", "recall_vs_best", "recall_vs_best_low", "recall_vs_best_high",
    "recall_vs_baseline", "recall_vs_baseline_low", "recall_vs_baseline_high",
    "precision_2000tok", "full_support_2000tok",
    "ndcg@10", "ndcg_vs_best", "ndcg_vs_best_low", "ndcg_vs_best_high",
    "full_support@10", "full_vs_best", "full_vs_best_low", "full_vs_best_high",
    "mrr@10", "recall@10", "recall@50",
    *[f"{label}_{suffix}" for label in TYPES.values()
      for suffix in ("recall", "vs_best", "vs_best_low", "vs_best_high")],
    "refused", "repaired", "cut_short", "billed_usd",
]


def table(runs: dict[str, dict]) -> list[dict]:
    rows = []
    for run_id in ORDER:
        run = runs.get(run_id)
        if run is None:
            continue
        by_question = scored(run)
        metrics = run["metrics"]
        notes = [r.get("rerank", {}) for r in run["rows"]]
        row = {
            "run": run_id,
            "reranker": run["models"].get("reranker", ""),
            "depth": next((n["depth"] for n in notes if n), ""),
            "recall_2000tok": metrics[RECALL]["mean"],
            **diff(run, runs[HYBRID], RECALL, "recall_vs_best"),
            **diff(run, runs[BASELINE], RECALL, "recall_vs_baseline"),
            "precision_2000tok": metrics[PRECISION]["mean"],
            "full_support_2000tok": statistics.fmean(
                r["scores"][RECALL] == 1.0 for r in by_question.values()
            ),
            "ndcg@10": metrics[NDCG]["mean"],
            **diff(run, runs[HYBRID], NDCG, "ndcg_vs_best"),
            FULL: metrics[FULL]["mean"],
            **diff(run, runs[HYBRID], FULL, "full_vs_best"),
            "mrr@10": metrics["mrr@10"]["mean"],
            "recall@10": metrics["recall@10"]["mean"],
            DEEP: statistics.fmean(r["scores"][DEEP] for r in by_question.values()),
            # An LLM reranker's replies: refused by the provider, repaired after parsing
            # (a number repeated, missing or out of range), or cut off by the output limit.
            "refused": sum(bool(n.get("refused")) for n in notes),
            "repaired": sum(
                any(n.get(key, 0) for key in ("invalid", "repeated", "missing")) for n in notes
            ),
            "cut_short": sum(n.get("stop_reason") == "length" for n in notes),
            "billed_usd": run["cost_usd"]["billed_this_run"],
        }
        for kind, label in TYPES.items():
            row[f"{label}_recall"] = run["by_type"][kind][RECALL]["mean"]
            row |= diff(run, runs[HYBRID], RECALL, f"{label}_vs_best", kind)
        rows.append(row)
    return rows


def markdown(rows: list[dict], runs: dict[str, dict]) -> str:
    lines = [
        "| run | recall in 2,000 tok | vs best so far | vs baseline | full support in 2,000 tok "
        "| nDCG@10 | vs best so far | full support@10 | vs best so far | MRR@10 | recall@10 "
        "| recall@50 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['run']} | {_pct(r['recall_2000tok'])} | {_diff(r, 'recall_vs_best')} "
            f"| {_diff(r, 'recall_vs_baseline')} | {_pct(r['full_support_2000tok'])} "
            f"| {_pct(r['ndcg@10'])} | {_diff(r, 'ndcg_vs_best')} | {_pct(r[FULL])} "
            f"| {_diff(r, 'full_vs_best')} | {_pct(r['mrr@10'])} | {_pct(r['recall@10'])} "
            f"| {_pct(r[DEEP])} |"
        )
    lines += ["", "Recall in 2,000 tokens by question type, with the paired difference from the "
              "best so far:", "", "| run | " + " | ".join(TYPES.values()) + " |",
              "|---|" + "---|" * len(TYPES)]
    for r in rows:
        cells = [f"{_pct(r[f'{label}_recall'])} {_diff(r, f'{label}_vs_best')}".strip()
                 for label in TYPES.values()]
        lines.append(f"| {r['run']} | " + " | ".join(cells) + " |")
    lines += ["", "| comparison | what it isolates | recall in 2,000 tok | nDCG@10 "
              "| full support@10 | recall@50 |", "|---|---|---|---|---|---|"]
    for a, b, what in PAIRS:
        if a in runs and b in runs:
            cells = [_diff(diff(runs[a], runs[b], metric, "d"), "d")
                     for metric in (RECALL, NDCG, FULL, DEEP)]
            lines.append(f"| {a} against {b} | {what} | " + " | ".join(cells) + " |")
    llm = [r for r in rows if r["refused"] or r["repaired"] or r["billed_usd"]]
    if llm:
        lines += ["", "| LLM reranker run | refused | repaired | cut short | billed |",
                  "|---|---|---|---|---|"]
        lines += [f"| {r['run']} | {r['refused']} | {r['repaired']} | {r['cut_short']} "
                  f"| ${r['billed_usd']:.2f} |" for r in llm]
    return "\n".join(lines)


def fusion_losses(runs: dict[str, dict]) -> str:
    """The Stage 5 questions where fusion left less evidence within 2,000 tokens than the
    better of dense and BM25 had alone: does reranking get it back?"""
    with open(WINS, newline="") as f:
        wins = {row["question_id"]: row for row in csv.DictReader(f)}
    lost = [qid for qid, row in wins.items() if float(row["hybrid_vs_better_one"]) < 0]
    better = {q: max(float(wins[q]["dense_recall"]), float(wins[q]["sparse_recall"])) for q in lost}
    lines = [f"The {len(lost)} questions where fusion left less evidence in 2,000 tokens than "
             f"the better single retriever ({_pct(statistics.fmean(better.values()))} for that "
             "retriever):", "", "| run | recall in 2,000 tok on these | back at or above the "
             "better retriever |", "|---|---|---|"]
    for run_id in ORDER:
        if run_id not in runs:
            continue
        rows = scored(runs[run_id])
        recall = [rows[q]["scores"][RECALL] for q in lost]
        back = sum(rows[q]["scores"][RECALL] >= better[q] - 1e-9 for q in lost)
        lines.append(f"| {run_id} | {_pct(statistics.fmean(recall))} | {back} |")
    return "\n".join(lines)


def timing() -> str:
    """Milliseconds per question for each local reranker: the score cache off, the model
    warm, the config's batch size, on this machine's GPU. Loading is timed apart."""
    pipeline = Pipeline(load_config("configs/stage05_retrieval/hybrid-rrf.yaml", PipelineConfig))
    pipeline.load()
    questions = [q.text for q in read_questions(QUESTIONS) if q.evidence][:102]
    first = [pipeline.first_stage(q, 100).candidates for q in questions]
    lines = ["| reranker | depth | median ms | 95th percentile ms | questions | load s "
             "| device | batch |", "|---|---|---|---|---|---|---|---|"]
    for key, plan in TIMED.items():
        spec = load_config(CONFIGS / f"{key}-100.yaml", PipelineConfig).reranker
        params = {k: v for k, v in spec.params.items() if k != "cache"}
        reranker = build("reranker", spec.name, **params)
        for i in (0, 1):  # load the weights and warm the GPU
            reranker.rerank(questions[i], first[i][:50])
        for depth, n in plan.items():
            times = []
            for question, found in zip(questions[2 : 2 + n], first[2 : 2 + n], strict=True):
                started = time.perf_counter()
                reranker.rerank(question, found[:depth])
                times.append(1000 * (time.perf_counter() - started))
            times.sort()
            lines.append(
                f"| {key} | {depth} | {statistics.median(times):.0f} "
                f"| {times[min(n - 1, int(0.95 * n))]:.0f} | {n} "
                f"| {reranker.scorer.load_seconds:.1f} | {reranker.scorer.device} "
                f"| {reranker.batch_size} |"
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--timing", action="store_true")
    args = parser.parse_args()
    runs = {r["run_id"]: r for r in report.load_runs(RUNS)}
    rows = table(runs)
    print(markdown(rows, runs))
    print("\n" + fusion_losses(runs))
    if args.timing:
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
