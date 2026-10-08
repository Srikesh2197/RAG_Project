"""Turn run directories into the committed results table.

`runs/` is local and ignored by git. `results/results.csv` and the table in the README
are rebuilt from it and committed, so the numbers in the repo always come from a run.
"""

import csv
import json
from pathlib import Path
from typing import Any

from ragbasics.eval.stats import Interval, paired_bootstrap

START, END = "<!-- results:start -->", "<!-- results:end -->"
BASELINE_RUN = "baseline"
# The best configuration so far: the run each stage changes one variable against.
# Updated when a stage records a new best config.
BEST_RUN = "retr-hybrid-rrf"

# (metric key, column heading). Shares are shown as percentages.
RETRIEVAL_COLUMNS = [
    ("recall@5", "recall@5"),
    ("full_support@5", "full support@5"),
    ("mrr@10", "MRR@10"),
    ("ndcg@10", "nDCG@10"),
    ("recall@2000tok", "recall in 2,000 tok"),
    ("precision@2000tok", "precision in 2,000 tok"),
]
# Retrieval metrics also shown as a paired difference against the baseline. These two are
# the ones that stay comparable when the chunk size changes.
PAIRED_RETRIEVAL = [
    ("recall@2000tok", "recall in 2,000 tok vs baseline"),
    ("precision@2000tok", "precision in 2,000 tok vs baseline"),
]
# And against the best run so far, for the metric a stage decides on.
PAIRED_BEST = [
    ("recall@2000tok", "recall in 2,000 tok vs best so far"),
]
ANSWER_COLUMNS = [
    ("correct", "correct"),
    ("correct_answerable", "correct, answerable"),
    ("abstain_null", "abstains on null"),
    ("false_abstain", "false abstentions"),
    ("faithful", "faithful"),
]
TYPE_LABELS = {
    "comparison_query": "comparison",
    "inference_query": "inference",
    "temporal_query": "temporal",
    "null_query": "null",
}


def load_run(run_dir: Path) -> dict[str, Any]:
    run = json.loads((run_dir / "metrics.json").read_text())
    with open(run_dir / "questions.jsonl") as f:
        run["rows"] = [json.loads(line) for line in f]
    return run


def load_runs(runs_dir: Path) -> list[dict[str, Any]]:
    runs = [load_run(p.parent) for p in sorted(runs_dir.glob("*/metrics.json"))]
    return sorted(runs, key=lambda run: run["created"])


def _both_scored(row: dict[str, Any], others: dict[str, Any], metric: str) -> bool:
    other = others.get(row["question_id"], {}).get("scores", {})
    return metric in row["scores"] and metric in other


def compare(a: dict[str, Any], b: dict[str, Any], metric: str, samples: int = 2000) -> Interval:
    """Paired difference a - b in `metric`, over the questions both runs scored."""
    b_rows = {row["question_id"]: row for row in b["rows"]}
    pairs = [
        (row["scores"][metric], b_rows[row["question_id"]]["scores"][metric], row["cluster"])
        for row in a["rows"]
        if _both_scored(row, b_rows, metric)
    ]
    return paired_bootstrap(
        [p[0] for p in pairs], [p[1] for p in pairs], [p[2] for p in pairs], samples
    )


def flips(a: dict[str, Any], b: dict[str, Any], metric: str = "correct") -> tuple[int, int]:
    """(questions whose score differs between the runs, questions both scored)."""
    b_rows = {row["question_id"]: row for row in b["rows"]}
    shared = [
        (row["scores"][metric], b_rows[row["question_id"]]["scores"][metric])
        for row in a["rows"]
        if _both_scored(row, b_rows, metric)
    ]
    return sum(x != y for x, y in shared), len(shared)


def _pct(metric: dict[str, Any] | None) -> str:
    if not metric or metric["n"] == 0:
        return ""
    # Small shares (precision is a few percent) need a decimal to show any interval.
    digits = 1 if metric["high"] < 0.1 else 0
    low, high = (f"{100 * metric[side]:.{digits}f}" for side in ("low", "high"))
    return f"{100 * metric['mean']:.1f} ({low}–{high})"


def _signed(interval: Interval, digits: int = 0) -> str:
    if interval.n == 0:
        return ""
    low, high = (f"{100 * side:+.{digits}f}" for side in (interval.low, interval.high))
    return f"{100 * interval.mean:+.{max(digits, 1)}f} ({low} to {high})"


def _versus(run: dict[str, Any], baseline: dict[str, Any] | None, metric: str, digits: int) -> str:
    """Paired difference from the baseline run, or blank for the baseline itself."""
    if not baseline or run is baseline or metric not in baseline["metrics"]:
        return ""
    return _signed(compare(run, baseline, metric), digits)


def _markdown(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def results_markdown(runs: list[dict[str, Any]]) -> str:
    """Three tables: retrieval, answers, and answer correctness by question type."""
    baseline = next((run for run in runs if run["run_id"] == BASELINE_RUN), None)
    best = next((run for run in runs if run["run_id"] == BEST_RUN), None)
    # A repeat (sample > 0) redraws only the answers; its retrieval is the same run's.
    retrieval = [run for run in runs if "recall@5" in run["metrics"] and not run["sample"]]
    answered = [run for run in runs if "correct" in run["metrics"]]
    parts = [
        "Percentages with 95% intervals from redrawing evidence clusters. "
        "\"vs baseline\" is the paired difference on the same questions; an interval that "
        "excludes 0 is a difference the question sample does not explain."
    ]
    if best:
        parts[0] += (
            f" \"vs best so far\" is the same difference against `{BEST_RUN}`, the "
            "configuration the current stage changes one variable in."
        )
    if retrieval:
        n = retrieval[0]["metrics"]["recall@5"]["n"]
        parts += [
            f"**Retrieval** ({n} development questions with gold evidence)",
            _markdown(
                [
                    "run",
                    *[heading for _, heading in RETRIEVAL_COLUMNS],
                    *[heading for _, heading in PAIRED_RETRIEVAL],
                    *[heading for _, heading in PAIRED_BEST if best],
                ],
                [
                    [
                        run["run_id"],
                        *[_pct(run["metrics"].get(k)) for k, _ in RETRIEVAL_COLUMNS],
                        # Precision is a few percent, so its difference needs two decimals.
                        *[
                            _versus(run, baseline, k, 2 if k.startswith("precision") else 1)
                            for k, _ in PAIRED_RETRIEVAL
                        ],
                        *[_versus(run, best, k, 1) for k, _ in PAIRED_BEST if best],
                    ]
                    for run in retrieval
                ],
            ),
        ]
    if answered:
        rows = []
        for run in answered:
            delta = _versus(run, baseline, "correct", 0)
            cost = run["cost_usd"]["list_price"]
            rows.append(
                [
                    run["run_id"],
                    str(run["metrics"]["correct"]["n"]),
                    *[_pct(run["metrics"].get(key)) for key, _ in ANSWER_COLUMNS],
                    delta,
                    f"${cost:.2f}" if cost else "",
                ]
            )
        types = [t for t in TYPE_LABELS if any(t in run["by_type"] for run in answered)]
        parts += [
            "**Answers**",
            _markdown(
                ["run", "n", *[heading for _, heading in ANSWER_COLUMNS], "correct vs baseline",
                 "cost"],
                rows,
            ),
            "**Answer correctness by question type**",
            _markdown(
                ["run", *[TYPE_LABELS[t] for t in types]],
                [
                    [run["run_id"], *[_pct(run["by_type"][t].get("correct")) for t in types]]
                    for run in answered
                ],
            ),
        ]
    return "\n\n".join(parts)


def write_csv(runs: list[dict[str, Any]], path: Path) -> None:
    """One row per run and metric, overall and per question type."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run", "mode", "question_type", "metric", "mean", "low", "high", "n"])
        for run in runs:
            groups = {"all": run["metrics"], **run["by_type"]}
            for group, metrics in groups.items():
                for name, m in metrics.items():
                    writer.writerow(
                        [run["run_id"], run["mode"], group, name,
                         f"{m['mean']:.4f}", f"{m['low']:.4f}", f"{m['high']:.4f}", m["n"]]
                    )


def update_readme(readme: Path, table: str) -> bool:
    """Replace the text between the results markers. False if the markers are missing."""
    text = readme.read_text()
    if START not in text or END not in text:
        return False
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    readme.write_text(f"{head}{START}\n{table}\n{END}{tail}")
    return True
