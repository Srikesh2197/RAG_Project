"""Stage 4 sweep table: every embedder against the best config so far (recursive 128 with
text-embedding-3-small) and against the Stage 1 baseline, from runs/ and the index
directories.

    python scripts/stage04_sweep.py            # print the tables
    python scripts/stage04_sweep.py --write    # also write results/stage04_embedding.csv
                                               # and results/stage04_dimensions.svg
    python scripts/stage04_sweep.py --timing   # also time a query embedding per model
                                               # (loads the local models; the two OpenAI
                                               # models cost a fraction of a cent)

Every run here uses the same chunks, so recall@k and nDCG@10 are comparable between
rows as well as the token-budget metrics.
"""

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from ragbasics.config import PipelineConfig, load_config
from ragbasics.costs import EMBEDDING_PRICE
from ragbasics.eval import report
from ragbasics.eval.dataset import read_questions
from ragbasics.pipeline import Pipeline

RUNS = Path("runs")
INDEXES = Path("data/index")
CONFIGS = Path("configs/stage04_embedding")
QUESTIONS = Path("data/processed/questions_dev.jsonl")
CSV = Path("results/stage04_embedding.csv")
PLOT = Path("results/stage04_dimensions.svg")
BASELINE, BEST = "baseline", "chunk-recursive-128"
RECALL, PRECISION, NDCG = "recall@2000tok", "precision@2000tok", "ndcg@10"
# Row order: models at full size, then truncated lengths, then the prefix switched off.
ORDER = [
    BEST, "embed-3-large", "embed-qwen3", "embed-bge-small",
    "embed-3-small-512", "embed-3-small-256",
    "embed-3-large-1024", "embed-3-large-256",
    "embed-qwen3-256", "embed-bge-small-96",
    "embed-qwen3-noprefix", "embed-bge-small-noprefix",
]
# Full-size run for each truncated or prefix-less run, for the paired difference.
PARENT = {
    "embed-3-small-512": BEST, "embed-3-small-256": BEST,
    "embed-3-large-1024": "embed-3-large", "embed-3-large-256": "embed-3-large",
    "embed-qwen3-256": "embed-qwen3", "embed-bge-small-96": "embed-bge-small",
    "embed-qwen3-noprefix": "embed-qwen3", "embed-bge-small-noprefix": "embed-bge-small",
}
# Full-size runs whose vectors were computed on this machine or billed by the API. The
# other rows were built from the cache, so their build time says nothing.
BUILT = {BEST, "embed-3-large", "embed-qwen3", "embed-bge-small"}
THROUGHPUT_SAMPLE = 1000  # chunks embedded to measure a local model's speed
FIELDS = [
    "run", "model", "dimensions", "query_prefix", "usd_per_million_tokens",
    "index_mb", "index_seconds", "chunks_per_second", "search_ms", "query_embed_ms",
    "recall_2000tok", "recall_low", "recall_high",
    "recall_vs_best", "recall_vs_best_low", "recall_vs_best_high",
    "recall_vs_baseline", "recall_vs_baseline_low", "recall_vs_baseline_high",
    "recall_vs_parent", "recall_vs_parent_low", "recall_vs_parent_high",
    "precision_2000tok", "full_support_2000tok",
    "ndcg@10", "ndcg_vs_best", "ndcg_vs_best_low", "ndcg_vs_best_high",
    "recall@10", "recall@50", "mrr@10",
]


def index_stats(name: str, rng: np.random.Generator) -> dict:
    directory = INDEXES / name
    meta = json.loads((directory / "meta.json").read_text())["embedder"]["params"]
    vectors = np.load(directory / "vectors.npy")
    ingest = directory / "ingest.json"
    built = json.loads(ingest.read_text()) if ingest.exists() else {}
    # Exact search is one matrix-vector product, so its time follows the dimensions.
    queries = rng.standard_normal((50, vectors.shape[1])).astype(np.float32)
    started = time.perf_counter()
    for query in queries:
        np.argsort(-(vectors @ query), kind="stable")[:50]
    search_ms = 1000 * (time.perf_counter() - started) / len(queries)
    return {
        "model": meta["model"],
        "dimensions": vectors.shape[1],
        "query_prefix": bool(meta.get("query_prefix")),
        "usd_per_million_tokens": EMBEDDING_PRICE.get(meta["model"], 0.0),
        "index_mb": (directory / "vectors.npy").stat().st_size / 1e6,
        "index_seconds": built.get("seconds", "") if name in BUILT else "",
        "search_ms": search_ms,
    }


def timings(run_id: str, questions: list[str]) -> dict:
    """Median time to embed one question and, for a local model, chunks embedded per
    second. The cache is off so the model is called; loading it is not timed."""
    stem = run_id.removeprefix("embed-")
    path = CONFIGS / f"{stem}.yaml"
    if run_id == BEST:
        path = Path("configs/stage03_chunking/recursive-128.yaml")
    cfg = load_config(path, PipelineConfig).model_copy(update={"embedding_cache": None})
    embedder = Pipeline(cfg).embedder
    embedder.embed(questions[:2], kind="query")  # load the model, open the connection
    times = []
    for question in questions[2:]:
        started = time.perf_counter()
        embedder.embed([question], kind="query")
        times.append(time.perf_counter() - started)
    out = {"query_embed_ms": 1000 * statistics.median(times)}
    if cfg.embedder.name == "local":
        with open(INDEXES / BEST / "chunks.jsonl") as f:
            texts = [json.loads(line)["text"] for line in f][::15][:THROUGHPUT_SAMPLE]
        started = time.perf_counter()
        embedder.embed(texts)
        out["chunks_per_second"] = len(texts) / (time.perf_counter() - started)
    return out


def diff(run: dict, other: dict | None, metric: str, prefix: str) -> dict:
    if other is None or other is run:
        return {}
    d = report.compare(run, other, metric)
    return {prefix: d.mean, f"{prefix}_low": d.low, f"{prefix}_high": d.high}


def table(timing: bool) -> list[dict]:
    runs = {r["run_id"]: r for r in report.load_runs(RUNS)}
    rng = np.random.default_rng(0)
    questions = [q.text for q in read_questions(QUESTIONS)][:22]
    rows = []
    for run_id in ORDER:
        run = runs.get(run_id)
        if run is None:
            continue
        metrics = run["metrics"]
        scored = [r["scores"] for r in run["rows"] if RECALL in r["scores"]]
        row = {
            "run": run_id,
            **index_stats(run["name"], rng),
            "recall_2000tok": metrics[RECALL]["mean"],
            "recall_low": metrics[RECALL]["low"],
            "recall_high": metrics[RECALL]["high"],
            **diff(run, runs[BEST], RECALL, "recall_vs_best"),
            **diff(run, runs[BASELINE], RECALL, "recall_vs_baseline"),
            **diff(run, runs.get(PARENT.get(run_id, "")), RECALL, "recall_vs_parent"),
            "precision_2000tok": metrics[PRECISION]["mean"],
            "full_support_2000tok": statistics.mean(s[RECALL] == 1.0 for s in scored),
            "ndcg@10": metrics[NDCG]["mean"],
            **diff(run, runs[BEST], NDCG, "ndcg_vs_best"),
            "recall@10": metrics["recall@10"]["mean"],
            # Share of gold sentences anywhere in the 50 retrieved chunks: what a reranker
            # (Stage 6) would have to work with.
            "recall@50": statistics.mean(
                rank is not None
                for r in run["rows"] for rank in r.get("gold_ranks", [])
            ),
            "mrr@10": metrics["mrr@10"]["mean"],
        }
        if timing and run_id in BUILT:
            row |= timings(run_id, questions)
        rows.append(row)
    return rows


def _pct(x: float, digits: int = 1) -> str:
    return f"{100 * x:.{digits}f}"


def _diff(row: dict, key: str) -> str:
    if key not in row:
        return ""
    return (f"{100 * row[key]:+.1f} ({100 * row[f'{key}_low']:+.1f} to "
            f"{100 * row[f'{key}_high']:+.1f})")


def markdown(rows: list[dict]) -> str:
    lines = [
        "| run | dimensions | recall in 2,000 tok | vs best so far | vs baseline "
        "| precision in 2,000 tok | full support in 2,000 tok | nDCG@10 | nDCG@10 vs best so far "
        "| recall@10 | recall@50 | MRR@10 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['run']} | {r['dimensions']:,} | {_pct(r['recall_2000tok'])} "
            f"| {_diff(r, 'recall_vs_best')} | {_diff(r, 'recall_vs_baseline')} "
            f"| {_pct(r['precision_2000tok'], 2)} | {_pct(r['full_support_2000tok'])} "
            f"| {_pct(r['ndcg@10'])} | {_diff(r, 'ndcg_vs_best')} "
            f"| {_pct(r['recall@10'])} | {_pct(r['recall@50'])} | {_pct(r['mrr@10'])} |"
        )
    lines += [
        "",
        "| run | recall in 2,000 tok | vs the same model at full size, with its prefix |",
        "|---|---|---|",
    ]
    lines += [
        f"| {r['run']} | {_pct(r['recall_2000tok'])} | {_diff(r, 'recall_vs_parent')} |"
        for r in rows
        if "recall_vs_parent" in r
    ]
    lines += [
        "",
        "| run | model | $ per million tokens | index MB | build seconds | chunks per second "
        "| search ms | query embedding ms |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        embed_ms = f"{r['query_embed_ms']:.0f}" if "query_embed_ms" in r else ""
        speed = f"{r['chunks_per_second']:.0f}" if "chunks_per_second" in r else ""
        lines.append(
            f"| {r['run']} | {r['model']} | {r['usd_per_million_tokens']:.2f} "
            f"| {r['index_mb']:.0f} | {r['index_seconds']} | {speed} | {r['search_ms']:.1f} "
            f"| {embed_ms} |"
        )
    return "\n".join(lines)


def plot(rows: list[dict]) -> str:
    """Recall in 2,000 tokens against vector length, one line per model, as an SVG."""
    series: dict[str, list[dict]] = {}
    for r in rows:
        if r["run"].endswith("-noprefix"):
            continue
        series.setdefault(r["model"], []).append(r)
    colours = ["#1f5fa8", "#c2571a", "#2e8540", "#7a3e9d"]
    width, height, left, top, right, bottom = 680, 400, 60, 30, 520, 340
    ticks = [96, 256, 384, 512, 1024, 1536, 3072]
    low, high = np.log2(ticks[0]), np.log2(ticks[-1])

    def x(dimensions: int) -> float:
        return left + (right - left) * (np.log2(dimensions) - low) / (high - low)

    def y(share: float) -> float:
        return bottom - (bottom - top) * (100 * share - 30) / 50  # axis from 30% to 80%

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        'font-family="sans-serif" font-size="12">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        f'<text x="{left}" y="18" font-size="14">Evidence recall within 2,000 retrieved '
        "tokens, by vector length</text>",
    ]
    for tick in range(30, 81, 10):
        ty = y(tick / 100)
        out.append(f'<line x1="{left}" y1="{ty}" x2="{right}" y2="{ty}" stroke="#ddd"/>')
        out.append(f'<text x="{left - 8}" y="{ty + 4}" text-anchor="end">{tick}%</text>')
    for tick in ticks:
        out.append(f'<text x="{x(tick):.1f}" y="{bottom + 18}" text-anchor="middle">'
                   f"{tick:,}</text>")
    for row, (colour, (model, points)) in enumerate(zip(colours, series.items(), strict=False)):
        points = sorted(points, key=lambda r: r["dimensions"])
        path = " ".join(f"{x(r['dimensions']):.1f},{y(r['recall_2000tok']):.1f}" for r in points)
        out.append(f'<polyline points="{path}" fill="none" stroke="{colour}" stroke-width="2"/>')
        for r in points:
            px, py = x(r["dimensions"]), y(r["recall_2000tok"])
            out.append(f'<line x1="{px:.1f}" y1="{y(r["recall_low"]):.1f}" x2="{px:.1f}" '
                       f'y2="{y(r["recall_high"]):.1f}" stroke="{colour}" opacity="0.5"/>')
            out.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4" fill="{colour}"/>')
        # A legend to the right: three of the lines end within a point of each other.
        ly = top + 20 + 20 * row
        out.append(f'<circle cx="{right + 20}" cy="{ly - 4}" r="4" fill="{colour}"/>')
        out.append(f'<text x="{right + 30}" y="{ly}" fill="{colour}">'
                   f'{model.split("/")[-1]}</text>')
    out.append(f'<text x="{(left + right) / 2}" y="{bottom + 42}" text-anchor="middle">vector '
               "length (dimensions kept); bars are 95% intervals over evidence clusters</text>")
    out.append("</svg>")
    return "\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--timing", action="store_true")
    args = parser.parse_args()
    load_dotenv()  # the OpenAI timing needs the key
    rows = table(args.timing)
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
