"""Stage 3 sweep table: every chunking run against the baseline, from runs/ and the
index directories. Needs no API key.

    python scripts/stage03_sweep.py            # print the table
    python scripts/stage03_sweep.py --write    # also write results/stage03_chunking.csv
                                               # and results/stage03_chunk_size.svg
"""

import argparse
import csv
import json
import statistics
from pathlib import Path

import tiktoken

from ragbasics.eval import report
from ragbasics.eval.dataset import read_questions

RUNS = Path("runs")
INDEXES = Path("data/index")
QUESTIONS = Path("data/processed/questions_dev.jsonl")
CSV = Path("results/stage03_chunking.csv")
PLOT = Path("results/stage03_chunk_size.svg")
RECALL, PRECISION = "recall@2000tok", "precision@2000tok"
# Row order: the baseline, then one group per question the sweep asks.
ORDER = [
    "baseline", "chunk-sentence-512", "chunk-recursive-512",
    "chunk-recursive-1024", "chunk-recursive-256", "chunk-recursive-128", "chunk-recursive-64",
    "chunk-fixed-128",
    "chunk-fixed-512-overlap-10", "chunk-fixed-512-overlap-20",
    "chunk-semantic-128",
    "chunk-parent-child-128-512", "chunk-parent-child-128-256",
]
FIELDS = [
    "run", "chunks", "mean_tokens", "embedded_tokens", "index_seconds", "index_cost_usd",
    "gold_sentences_cut", "returned_in_budget",
    "recall_2000tok", "recall_low", "recall_high",
    "recall_vs_baseline", "recall_diff_low", "recall_diff_high",
    "precision_2000tok", "precision_vs_baseline", "precision_diff_low", "precision_diff_high",
    "full_support_2000tok", "recall@5", "mrr@10",
]


def index_stats(name: str, spans: set, encoding) -> dict:
    """Chunk count, mean size of the embedded text, and gold sentences no chunk holds whole."""
    directory = INDEXES / name
    with open(directory / "chunks.jsonl") as f:
        chunks = [json.loads(line) for line in f]
    sizes = [
        len(encoding.encode(c["embed_text"] or c["text"], disallowed_special=())) for c in chunks
    ]
    by_doc: dict[str, set] = {}
    for c in chunks:
        by_doc.setdefault(c["doc_id"], set()).add((c["start_char"], c["end_char"]))
    cut = sum(
        not any(a <= s.start_char and s.end_char <= b for a, b in by_doc[s.doc_id]) for s in spans
    )
    ingest = directory / "ingest.json"
    built = json.loads(ingest.read_text()) if ingest.exists() else {}
    return {
        "chunks": len(chunks),
        "mean_tokens": round(statistics.mean(sizes)),
        "embedded_tokens": built.get("embed_tokens", sum(sizes)) + built.get("chunker_tokens", 0),
        "index_seconds": built.get("seconds", ""),
        "index_cost_usd": built.get("cost_usd", ""),
        "gold_sentences_cut": cut,
    }


def table() -> list[dict]:
    runs = {r["run_id"]: r for r in report.load_runs(RUNS)}
    baseline = runs["baseline"]
    spans = {s for q in read_questions(QUESTIONS) for s in q.evidence}
    encoding = tiktoken.get_encoding("cl100k_base")
    rows = []
    for run_id in ORDER:
        run = runs.get(run_id)
        if run is None:
            continue
        recall, precision = run["metrics"][RECALL], run["metrics"][PRECISION]
        d_recall = report.compare(run, baseline, RECALL)
        d_precision = report.compare(run, baseline, PRECISION)
        scored = [r["scores"] for r in run["rows"] if RECALL in r["scores"]]
        rows.append(
            {
                "run": run_id,
                **index_stats(run["name"], spans, encoding),
                "recall_2000tok": recall["mean"],
                "recall_low": recall["low"],
                "recall_high": recall["high"],
                "recall_vs_baseline": d_recall.mean,
                "recall_diff_low": d_recall.low,
                "recall_diff_high": d_recall.high,
                "precision_2000tok": precision["mean"],
                "precision_vs_baseline": d_precision.mean,
                "precision_diff_low": d_precision.low,
                "precision_diff_high": d_precision.high,
                # Every gold sentence inside the budget: what a multi-hop answer needs.
                "full_support_2000tok": statistics.mean(s[RECALL] == 1.0 for s in scored),
                "recall@5": run["metrics"]["recall@5"]["mean"],
                "mrr@10": run["metrics"]["mrr@10"]["mean"],
            }
        )
    return rows


def markdown(rows: list[dict]) -> str:
    def pct(x: float, digits: int = 1) -> str:
        return f"{100 * x:.{digits}f}"

    def diff(row: dict, key: str, digits: int) -> str:
        if row["run"] == "baseline":
            return ""
        mean, low, high = (100 * row[f"{key}{s}"] for s in ("_vs_baseline", "_diff_low",
                                                              "_diff_high"))
        return f"{mean:+.{digits}f} ({low:+.{digits}f} to {high:+.{digits}f})"

    lines = [
        "| run | chunks | mean tokens | gold sentences cut | recall in 2,000 tok | vs baseline "
        "| precision | vs baseline | full support in 2,000 tok | index seconds |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['run']} | {r['chunks']:,} | {r['mean_tokens']} | {r['gold_sentences_cut']} "
            f"| {pct(r['recall_2000tok'])} | {diff(r, 'recall', 1)} "
            f"| {pct(r['precision_2000tok'], 2)} | {diff(r, 'precision', 2)} "
            f"| {pct(r['full_support_2000tok'])} | {r['index_seconds']} |"
        )
    return "\n".join(lines)


def plot(rows: list[dict]) -> str:
    """Recall in 2,000 tokens against chunk size for the recursive chunker, as an SVG."""
    points = sorted(
        (int(r["run"].rsplit("-", 1)[1]), r)
        for r in rows
        if r["run"].startswith("chunk-recursive-")
    )
    base = next(r for r in rows if r["run"] == "baseline")
    width, height, left, top, right, bottom = 640, 380, 60, 30, 610, 320
    sizes = [size for size, _ in points]

    def x(size: int) -> float:  # sizes double each step, so they are spaced evenly
        return left + (right - left) * sizes.index(size) / (len(sizes) - 1)

    def y(share: float) -> float:
        return bottom - (bottom - top) * (100 * share - 20) / 60  # axis from 20% to 80%

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        'font-family="sans-serif" font-size="12">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        f'<text x="{left}" y="18" font-size="14">Evidence recall within 2,000 retrieved '
        "tokens, by chunk size (recursive chunker)</text>",
    ]
    for tick in range(20, 81, 10):
        ty = y(tick / 100)
        out.append(f'<line x1="{left}" y1="{ty}" x2="{right}" y2="{ty}" stroke="#ddd"/>')
        out.append(f'<text x="{left - 8}" y="{ty + 4}" text-anchor="end">{tick}%</text>')
    by = y(base["recall_2000tok"])
    out.append(f'<line x1="{left}" y1="{by}" x2="{right}" y2="{by}" stroke="#888" '
               'stroke-dasharray="5 4"/>')
    out.append(f'<text x="{right}" y="{by - 6}" text-anchor="end" fill="#555">baseline: fixed '
               f'512, {100 * base["recall_2000tok"]:.1f}%</text>')
    path = " ".join(f"{x(s):.1f},{y(r['recall_2000tok']):.1f}" for s, r in points)
    out.append(f'<polyline points="{path}" fill="none" stroke="#1f5fa8" stroke-width="2"/>')
    for size, r in points:
        px = x(size)
        out.append(f'<line x1="{px}" y1="{y(r["recall_low"]):.1f}" x2="{px}" '
                   f'y2="{y(r["recall_high"]):.1f}" stroke="#1f5fa8"/>')
        out.append(f'<circle cx="{px}" cy="{y(r["recall_2000tok"]):.1f}" r="4" fill="#1f5fa8"/>')
        out.append(f'<text x="{px + 8}" y="{y(r["recall_2000tok"]) - 8:.1f}">'
                   f'{100 * r["recall_2000tok"]:.1f}</text>')
        out.append(f'<text x="{px}" y="{bottom + 18}" text-anchor="middle">{size}</text>')
    out.append(f'<text x="{(left + right) / 2}" y="{bottom + 40}" text-anchor="middle">chunk '
               "size in tokens; bars are 95% intervals over evidence clusters</text>")
    out.append("</svg>")
    return "\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    rows = table()
    print(markdown(rows))
    if args.write:
        with open(CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({k: round(v, 4) if isinstance(v, float) else v
                                 for k, v in row.items()})
        PLOT.write_text(plot(rows))
        print(f"\nWrote {CSV} and {PLOT}")


if __name__ == "__main__":
    main()
