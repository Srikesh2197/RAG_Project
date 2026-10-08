"""Stage 5 tables: dense, sparse and hybrid retrieval over the same chunks, from runs/.

    python scripts/stage05_sweep.py            # print the tables
    python scripts/stage05_sweep.py --write    # also write results/stage05_retrieval.csv,
                                               # stage05_wins.csv, stage05_examples.md and
                                               # stage05_fusion_weight.svg
    python scripts/stage05_sweep.py --timing   # also time index builds and queries
    python scripts/stage05_sweep.py --parity   # also compare with bm25s on the corpus

Every run ranks the 15,149 chunks of data/index/chunk-recursive-128, so recall@k, MRR
and nDCG@10 are comparable between rows as well as the token-budget metrics.

The fusion plot reads results/stage05_tuning.csv, written by scripts/stage05_tune.py.
"""

import argparse
import csv
import statistics
import time
from pathlib import Path

from ragbasics.config import PipelineConfig, load_config
from ragbasics.eval import report
from ragbasics.eval.dataset import read_documents, read_questions
from ragbasics.eval.stats import paired_bootstrap
from ragbasics.index.bm25 import BM25Index
from ragbasics.pipeline import Pipeline
from ragbasics.retrieval.sparse import SparseRetriever
from ragbasics.retrieval.tokenize import Tokenizer

RUNS = Path("runs")
CONFIGS = Path("configs/stage05_retrieval")
QUESTIONS = Path("data/processed/questions_dev.jsonl")
DOCUMENTS = Path("data/processed/documents.jsonl")
TUNING = Path("results/stage05_tuning.csv")
CSV = Path("results/stage05_retrieval.csv")
WINS = Path("results/stage05_wins.csv")
EXAMPLES = Path("results/stage05_examples.md")
PLOT = Path("results/stage05_fusion_weight.svg")

BASELINE, DENSE = "baseline", "chunk-recursive-128"
SPARSE, HYBRID = "retr-sparse", "retr-hybrid-rrf"
ORDER = [
    DENSE, "retr-sparse-default", SPARSE, HYBRID, "retr-hybrid-rrf-k10",
    "retr-hybrid-weighted", "retr-sparse-header", "retr-hybrid-rrf-k10-header",
]
# Extra comparisons: (run, against, what the difference isolates).
PAIRS = [
    (SPARSE, "retr-sparse-default", "stopwords, stemming, k1 and b chosen on the pool"),
    (HYBRID, SPARSE, "adding dense to BM25"),
    ("retr-hybrid-rrf-k10", HYBRID, "RRF constant 10 instead of 60"),
    ("retr-hybrid-weighted", HYBRID, "weighted fusion instead of RRF at 60"),
    ("retr-hybrid-weighted", "retr-hybrid-rrf-k10", "weighted fusion instead of RRF at 10"),
    ("retr-sparse-header", SPARSE, "title, source and date in the BM25 index"),
    ("retr-hybrid-rrf-k10-header", "retr-hybrid-rrf-k10", "the same header, in the hybrid"),
]
RECALL, PRECISION, NDCG = "recall@2000tok", "precision@2000tok", "ndcg@10"
DEEP = "recall@50"
TYPES = {"comparison_query": "comparison", "inference_query": "inference",
         "temporal_query": "temporal"}
EXAMPLES_PER_TYPE = 5
FIELDS = [
    "run", "retriever",
    "recall_2000tok", "recall_low", "recall_high",
    "recall_vs_best", "recall_vs_best_low", "recall_vs_best_high",
    "recall_vs_baseline", "recall_vs_baseline_low", "recall_vs_baseline_high",
    "precision_2000tok", "full_support_2000tok",
    "ndcg@10", "ndcg_vs_best", "ndcg_vs_best_low", "ndcg_vs_best_high",
    "mrr@10", "recall@5", "recall@10",
    "recall@50", "recall@50_vs_best", "recall@50_vs_best_low", "recall@50_vs_best_high",
    *[f"{label}_{suffix}" for label in TYPES.values()
      for suffix in ("recall", "vs_best", "vs_best_low", "vs_best_high")],
]


def scored(run: dict) -> dict[str, dict]:
    """question id -> row, for the questions with gold evidence. Adds recall@50: the
    share of the question's gold sentences anywhere in the 50 retrieved chunks, which is
    what a reranker (Stage 6) has to work with."""
    rows = {}
    for row in run["rows"]:
        if "gold_ranks" in row:
            row["scores"][DEEP] = statistics.fmean(r is not None for r in row["gold_ranks"])
            rows[row["question_id"]] = row
    return rows


def diff(run: dict, other: dict, metric: str, prefix: str, kind: str | None = None) -> dict:
    """Paired difference run - other in `metric`, optionally for one question type."""
    if run is other:
        return {}
    a, b = scored(run), scored(other)
    ids = [q for q in a if q in b and (kind is None or a[q]["question_type"] == kind)]
    d = paired_bootstrap(
        [a[q]["scores"][metric] for q in ids],
        [b[q]["scores"][metric] for q in ids],
        [a[q]["cluster"] for q in ids],
    )
    return {prefix: d.mean, f"{prefix}_low": d.low, f"{prefix}_high": d.high}


def table(runs: dict[str, dict]) -> list[dict]:
    rows = []
    for run_id in ORDER:
        run = runs.get(run_id)
        if run is None:
            continue
        by_question = scored(run)
        metrics = run["metrics"]
        row = {
            "run": run_id,
            "retriever": load_retriever(run_id),
            "recall_2000tok": metrics[RECALL]["mean"],
            "recall_low": metrics[RECALL]["low"],
            "recall_high": metrics[RECALL]["high"],
            **diff(run, runs[DENSE], RECALL, "recall_vs_best"),
            **diff(run, runs[BASELINE], RECALL, "recall_vs_baseline"),
            "precision_2000tok": metrics[PRECISION]["mean"],
            "full_support_2000tok": statistics.fmean(
                r["scores"][RECALL] == 1.0 for r in by_question.values()
            ),
            "ndcg@10": metrics[NDCG]["mean"],
            **diff(run, runs[DENSE], NDCG, "ndcg_vs_best"),
            "mrr@10": metrics["mrr@10"]["mean"],
            "recall@5": metrics["recall@5"]["mean"],
            "recall@10": metrics["recall@10"]["mean"],
            DEEP: statistics.fmean(r["scores"][DEEP] for r in by_question.values()),
            **diff(run, runs[DENSE], DEEP, f"{DEEP}_vs_best"),
        }
        for kind, label in TYPES.items():
            row[f"{label}_recall"] = run["by_type"][kind][RECALL]["mean"]
            row |= diff(run, runs[DENSE], RECALL, f"{label}_vs_best", kind)
        rows.append(row)
    return rows


def load_retriever(run_id: str) -> str:
    if not run_id.startswith("retr-"):
        return "dense"
    cfg = load_config(CONFIGS / f"{run_id.removeprefix('retr-')}.yaml", PipelineConfig)
    return cfg.retriever.name


def _pct(x: float, digits: int = 1) -> str:
    return f"{100 * x:.{digits}f}"


def _diff(row: dict, key: str) -> str:
    if key not in row:
        return ""
    return (f"{100 * row[key]:+.1f} ({100 * row[f'{key}_low']:+.1f} to "
            f"{100 * row[f'{key}_high']:+.1f})")


def markdown(rows: list[dict], runs: dict[str, dict]) -> str:
    lines = [
        "| run | recall in 2,000 tok | vs best so far | vs baseline | precision in 2,000 tok "
        "| full support in 2,000 tok | nDCG@10 | nDCG@10 vs best so far | MRR@10 | recall@10 "
        "| recall@50 | recall@50 vs best so far |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['run']} | {_pct(r['recall_2000tok'])} | {_diff(r, 'recall_vs_best')} "
            f"| {_diff(r, 'recall_vs_baseline')} | {_pct(r['precision_2000tok'], 2)} "
            f"| {_pct(r['full_support_2000tok'])} | {_pct(r['ndcg@10'])} "
            f"| {_diff(r, 'ndcg_vs_best')} | {_pct(r['mrr@10'])} | {_pct(r['recall@10'])} "
            f"| {_pct(r[DEEP])} | {_diff(r, f'{DEEP}_vs_best')} |"
        )
    lines += [
        "",
        "Recall in 2,000 tokens by question type, with the paired difference from the best "
        "so far:",
        "",
        "| run | " + " | ".join(TYPES.values()) + " |",
        "|---|" + "---|" * len(TYPES),
    ]
    for r in rows:
        cells = [
            f"{_pct(r[f'{label}_recall'])} {_diff(r, f'{label}_vs_best')}".strip()
            for label in TYPES.values()
        ]
        lines.append(f"| {r['run']} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "| comparison | what it isolates | recall in 2,000 tok | nDCG@10 | recall@50 |",
        "|---|---|---|---|---|",
    ]
    for a, b, what in PAIRS:
        if a in runs and b in runs:
            cells = [
                _diff(diff(runs[a], runs[b], metric, "d"), "d") for metric in (RECALL, NDCG, DEEP)
            ]
            lines.append(f"| {a} against {b} | {what} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


# --- Which retriever found what -------------------------------------------------------------


def gold_sentences(runs: dict[str, dict]) -> list[dict]:
    """One record per (question, gold sentence): its text and its rank in the dense, the
    sparse and the hybrid list (None when it is not in the top 50)."""
    dense, sparse, hybrid = (scored(runs[name]) for name in (DENSE, SPARSE, HYBRID))
    texts = {d.doc_id: d.text for d in read_documents(DOCUMENTS)}
    tokenizer = Tokenizer(stopwords=True, stemmer="porter")
    records = []
    for question in read_questions(QUESTIONS):
        if not question.evidence:
            continue
        asked = set(tokenizer(question.text))
        for i, span in enumerate(question.evidence):
            sentence = texts[span.doc_id][span.start_char : span.end_char]
            terms = set(tokenizer(sentence))
            records.append({
                "question_id": question.question_id,
                "question_type": question.question_type,
                "cluster": question.cluster,
                "question": question.text,
                "sentence": sentence,
                # Share of the sentence's terms that the question also uses.
                "overlap": len(terms & asked) / len(terms) if terms else 0.0,
                "dense": dense[question.question_id]["gold_ranks"][i],
                "sparse": sparse[question.question_id]["gold_ranks"][i],
                "hybrid": hybrid[question.question_id]["gold_ranks"][i],
            })
    return records


def found_by(records: list[dict], depth: int) -> dict[str, list[dict]]:
    """Gold sentences split by which of dense and sparse has them in the top `depth`."""
    groups: dict[str, list[dict]] = {"both": [], "dense only": [], "sparse only": [],
                                     "neither": []}
    for r in records:
        in_dense = r["dense"] is not None and r["dense"] <= depth
        in_sparse = r["sparse"] is not None and r["sparse"] <= depth
        key = ("both" if in_dense and in_sparse else "dense only" if in_dense
               else "sparse only" if in_sparse else "neither")
        groups[key].append(r)
    return groups


def overlap_markdown(records: list[dict]) -> str:
    lines = [
        "Gold sentences by which retriever has them, and how much of the sentence's "
        "wording the question repeats (median share of the sentence's terms that are also "
        "in the question, after stopword removal and stemming):",
        "",
        "| | in the top 20 | hybrid also has it in its top 20 | median overlap "
        "| in the top 50 |",
        "|---|---|---|---|---|",
    ]
    top20, top50 = found_by(records, 20), found_by(records, 50)
    for key in top20:
        group = top20[key]
        kept = sum(r["hybrid"] is not None and r["hybrid"] <= 20 for r in group)
        overlap = statistics.median(r["overlap"] for r in group) if group else 0.0
        lines.append(f"| {key} | {len(group)} | {kept} | {_pct(overlap, 0)}% "
                     f"| {len(top50[key])} |")
    lines.append(f"| all | {len(records)} | | "
                 f"{_pct(statistics.median(r['overlap'] for r in records), 0)}% "
                 f"| {len(records)} |")
    return "\n".join(lines)


def wins(runs: dict[str, dict]) -> list[dict]:
    """Per question: recall in 2,000 tokens for dense, sparse and hybrid, and who wins."""
    dense, sparse, hybrid = (scored(runs[name]) for name in (DENSE, SPARSE, HYBRID))
    rows = []
    for qid, row in dense.items():
        d, s, h = (r[qid]["scores"][RECALL] for r in (dense, sparse, hybrid))
        rows.append({
            "question_id": qid,
            "question_type": TYPES[row["question_type"]],
            "cluster": row["cluster"],
            "winner": "dense" if d > s else "sparse" if s > d else "tie",
            "dense_recall": round(d, 3),
            "sparse_recall": round(s, 3),
            "hybrid_recall": round(h, 3),
            "hybrid_vs_better_one": round(h - max(d, s), 3),
            "dense_gold_ranks": row["gold_ranks"],
            "sparse_gold_ranks": sparse[qid]["gold_ranks"],
            "hybrid_gold_ranks": hybrid[qid]["gold_ranks"],
            "question": row["question"],
        })
    return rows


def wins_markdown(rows: list[dict]) -> str:
    lines = [
        "Questions by which retriever finds more of the gold evidence within 2,000 tokens:",
        "",
        "| | all | " + " | ".join(TYPES.values()) + " |",
        "|---|---|" + "---|" * len(TYPES),
    ]
    outcomes = [
        ("sparse wins", lambda r: r["winner"] == "sparse"),
        ("dense wins", lambda r: r["winner"] == "dense"),
        ("tie, all evidence found by both", lambda r: r["winner"] == "tie"
         and r["dense_recall"] == 1),
        ("tie, some evidence missed by both", lambda r: r["winner"] == "tie"
         and r["dense_recall"] < 1),
        ("hybrid below the better of the two", lambda r: r["hybrid_vs_better_one"] < 0),
        ("hybrid above both", lambda r: r["hybrid_vs_better_one"] > 0),
    ]
    for label, test in outcomes:
        counts = [sum(test(r) for r in rows)] + [
            sum(test(r) for r in rows if r["question_type"] == kind) for kind in TYPES.values()
        ]
        lines.append(f"| {label} | " + " | ".join(str(c) for c in counts) + " |")
    lines.append(f"| questions | {len(rows)} | " + " | ".join(
        str(sum(r["question_type"] == kind for r in rows)) for kind in TYPES.values()) + " |")
    return "\n".join(lines)


def _rank(rank: int | None) -> str:
    return "not in 50" if rank is None else str(rank)


def examples(records: list[dict]) -> str:
    """Five gold sentences per failure type, at most one per evidence cluster and per
    sentence, taking the widest gaps first."""
    beyond = 51

    def gap(a: str, b: str):
        return lambda r: (r[b] or beyond) - (r[a] or beyond)

    failures = [
        ("Dense misses what BM25 finds",
         "BM25 has the sentence in its top 10; dense does not have it in its top 50.",
         lambda r: r["sparse"] is not None and r["sparse"] <= 10 and r["dense"] is None,
         lambda r: r["sparse"]),
        ("BM25 misses what dense finds",
         "Dense has the sentence in its top 10; BM25 does not have it in its top 50.",
         lambda r: r["dense"] is not None and r["dense"] <= 10 and r["sparse"] is None,
         lambda r: r["dense"]),
        ("Both miss",
         "Neither retriever has the sentence in its top 50. Lowest overlap first.",
         lambda r: r["dense"] is None and r["sparse"] is None,
         lambda r: r["overlap"]),
        ("Fusion loses what one retriever had",
         "One retriever has the sentence in its top 10 and the other not in its top 50; "
         "the fused list puts it below rank 20, outside a 2,000-token prompt.",
         lambda r: min(r["dense"] or beyond, r["sparse"] or beyond) <= 10
         and (r["dense"] is None or r["sparse"] is None)
         and (r["hybrid"] is None or r["hybrid"] > 20),
         lambda r: -(r["hybrid"] or beyond)),
    ]
    out = ["# Stage 5: example gold sentences for each failure type", "",
           "Written by `python scripts/stage05_sweep.py --write`. Ranks are of the first "
           f"chunk that overlaps the gold sentence, in `{DENSE}` (dense), `{SPARSE}` (BM25) "
           f"and `{HYBRID}` (hybrid). Overlap is the share of the sentence's terms that the "
           "question also uses.", ""]
    for title, rule, test, order in failures:
        matching = sorted((r for r in records if test(r)), key=order)
        out += [f"## {title}", "", f"{rule} {len(matching)} of {len(records)} gold "
                "sentences.", ""]
        clusters: set[str] = set()
        sentences: set[str] = set()
        shown = 0
        for r in matching:
            if r["cluster"] in clusters or r["sentence"] in sentences:
                continue
            clusters.add(r["cluster"])
            sentences.add(r["sentence"])
            shown += 1
            out += [
                f"**{shown}. {r['question_id']}, {TYPES[r['question_type']]}.** "
                f"Dense {_rank(r['dense'])}, BM25 {_rank(r['sparse'])}, hybrid "
                f"{_rank(r['hybrid'])}. Overlap {_pct(r['overlap'], 0)}%.",
                "",
                f"- Question: {r['question']}",
                f"- Gold sentence: {r['sentence'].strip()}",
                "",
            ]
            if shown == EXAMPLES_PER_TYPE:
                break
    return "\n".join(out)


# --- Speed, size and the library ------------------------------------------------------------


def timing() -> str:
    cfg = load_config(CONFIGS / "hybrid-rrf.yaml", PipelineConfig)
    pipeline = Pipeline(cfg)
    pipeline.load()
    chunks = pipeline.store.chunks
    lines = ["| BM25 index | terms | build seconds | on disk MB |", "|---|---|---|---|"]
    for settings in ({"stopwords": False, "stemmer": "none"},
                     {"stopwords": True, "stemmer": "porter"},
                     {"stopwords": True, "stemmer": "porter", "header": True}):
        retriever = SparseRetriever(**settings)
        started = time.perf_counter()
        index = BM25Index.build([retriever.tokenizer(retriever.indexed_text(c)) for c in chunks])
        seconds = time.perf_counter() - started
        directory = retriever.directory(pipeline.index_dir)
        size = sum(f.stat().st_size for f in directory.glob("*")) / 1e6
        lines.append(f"| {settings} | {len(index.vocabulary):,} | {seconds:.1f} | {size:.1f} |")
    questions = [q.text for q in read_questions(QUESTIONS)][:200]
    pipeline.retrieve(questions[0], 50)  # load the BM25 index and warm the cache
    steps: dict[str, list[float]] = {}
    for question in questions:
        for step, seconds in pipeline.retrieve(question, 50).timings.items():
            steps.setdefault(step, []).append(seconds)
    lines += ["", "Median milliseconds per question, hybrid with RRF, 200 chunks from each "
              "retriever (the question's vector comes from the cache, so `embed_query` is "
              "a lookup and not the API call):", ""]
    lines += [f"- {step}: {1000 * statistics.median(times):.2f}" for step, times in steps.items()]
    return "\n".join(lines)


def parity() -> str:
    """The hand-written index against bm25s on the corpus: same terms in, top 50 out."""
    import sys

    sys.path.insert(0, "framework_equivalents")
    import bm25_library

    cfg = load_config(CONFIGS / "sparse.yaml", PipelineConfig)
    pipeline = Pipeline(cfg)
    pipeline.load()
    retriever = pipeline.retriever
    mine = retriever.index_for(pipeline)
    documents = [retriever.tokenizer(retriever.indexed_text(c)) for c in pipeline.store.chunks]
    started = time.perf_counter()
    theirs = bm25_library.build(documents, retriever.k1, retriever.b)
    build = time.perf_counter() - started
    questions = [q.text for q in read_questions(QUESTIONS) if q.evidence]
    same_scores = same_order = same_set = 0
    seconds = {"hand-written": 0.0, "bm25s": 0.0}
    for question in questions:
        terms = retriever.tokenizer(question)
        t0 = time.perf_counter()
        ours = mine.search(terms, 50, retriever.k1, retriever.b)
        t1 = time.perf_counter()
        library = bm25_library.search(theirs, terms, 50)
        seconds["hand-written"] += t1 - t0
        seconds["bm25s"] += time.perf_counter() - t1
        same_scores += all(abs(a[1] - b[1]) < 1e-4 for a, b in zip(ours, library, strict=True))
        same_order += [i for i, _ in ours] == [i for i, _ in library]
        same_set += {i for i, _ in ours} == {i for i, _ in library}
    n = len(questions)
    return (
        f"bm25s against the hand-written index, {n} development questions, top 50:\n"
        f"- the same score at every rank: {same_scores} of {n}\n"
        f"- the same 50 chunks: {same_set} of {n}\n"
        f"- the same order: {same_order} of {n}\n"
        f"- bm25s index build {build:.1f} s; per query, hand-written "
        f"{1000 * seconds['hand-written'] / n:.2f} ms, bm25s {1000 * seconds['bm25s'] / n:.2f} ms"
    )


# --- Plot -----------------------------------------------------------------------------------


def plot() -> str:
    """Recall in 2,000 tokens against the dense weight, for both normalisers and both
    question sets, with reciprocal rank fusion as horizontal reference lines."""
    with open(TUNING, newline="") as f:
        rows = list(csv.DictReader(f))
    width, height, left, top, right, bottom = 800, 400, 60, 30, 500, 340
    colours = {"minmax": "#1f5fa8", "zscore": "#c2571a", "rrf": "#555555"}

    def x(weight: float) -> float:
        return left + (right - left) * weight

    def y(share: float) -> float:
        return bottom - (bottom - top) * (100 * share - 55) / 25  # axis from 55% to 80%

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        'font-family="sans-serif" font-size="12">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        f'<text x="{left}" y="18" font-size="14">Evidence recall within 2,000 retrieved '
        "tokens, by fusion weight</text>",
    ]
    for tick in range(55, 81, 5):
        out.append(f'<line x1="{left}" y1="{y(tick / 100)}" x2="{right}" y2="{y(tick / 100)}" '
                   'stroke="#ddd"/>')
        out.append(f'<text x="{left - 8}" y="{y(tick / 100) + 4}" text-anchor="end">'
                   f"{tick}%</text>")
    for tick in (0, 0.25, 0.5, 0.75, 1):
        out.append(f'<text x="{x(tick):.1f}" y="{bottom + 18}" text-anchor="middle">'
                   f"{tick:g}</text>")
    legend = []
    for split, dash, label in (("dev", "", "development (reported)"),
                               ("pool", ' stroke-dasharray="5 4"', "pool (tuned on)")):
        for normalise in ("minmax", "zscore"):
            points = sorted(
                (float(r["weight"]), float(r["recall_2000tok"])) for r in rows
                if (r["split"], r["family"], r["normalise"]) == (split, "weighted", normalise)
            )
            path = " ".join(f"{x(w):.1f},{y(v):.1f}" for w, v in points)
            out.append(f'<polyline points="{path}" fill="none" stroke="{colours[normalise]}" '
                       f'stroke-width="2"{dash}/>')
            if split == "dev":
                out += [f'<circle cx="{x(w):.1f}" cy="{y(v):.1f}" r="3" '
                        f'fill="{colours[normalise]}"/>' for w, v in points]
            legend.append((colours[normalise], dash, f"{normalise}, {label}"))
        rrf = next(float(r["recall_2000tok"]) for r in rows
                   if (r["split"], r["family"], r["rrf_k"]) == (split, "rrf", "60"))
        out.append(f'<line x1="{left}" y1="{y(rrf):.1f}" x2="{right}" y2="{y(rrf):.1f}" '
                   f'stroke="{colours["rrf"]}" stroke-width="1.5"{dash}/>')
        legend.append((colours["rrf"], dash, f"RRF at 60, {label}"))
    for row, (colour, dash, label) in enumerate(legend):
        ly = top + 20 + 20 * row
        out.append(f'<line x1="{right + 14}" y1="{ly - 4}" x2="{right + 40}" y2="{ly - 4}" '
                   f'stroke="{colour}" stroke-width="2"{dash}/>')
        out.append(f'<text x="{right + 46}" y="{ly}">{label}</text>')
    out.append(f'<text x="{x(0):.1f}" y="{bottom + 36}" text-anchor="middle">BM25 only</text>')
    out.append(f'<text x="{x(1):.1f}" y="{bottom + 36}" text-anchor="middle">dense only</text>')
    out.append(f'<text x="{(left + right) / 2}" y="{bottom + 52}" text-anchor="middle">'
               "weight of the dense list</text>")
    out.append("</svg>")
    return "\n".join(out) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--timing", action="store_true")
    parser.add_argument("--parity", action="store_true")
    args = parser.parse_args()
    runs = {r["run_id"]: r for r in report.load_runs(RUNS)}
    rows = table(runs)
    records = gold_sentences(runs)
    won = wins(runs)
    print(markdown(rows, runs))
    print("\n" + wins_markdown(won))
    print("\n" + overlap_markdown(records))
    if args.timing:
        print("\n" + timing())
    if args.parity:
        print("\n" + parity())
    if args.write:
        with open(CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore", restval="")
            writer.writeheader()
            for row in rows:
                writer.writerow({k: round(v, 4) if isinstance(v, float) else v
                                 for k, v in row.items()})
        with open(WINS, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(won[0]))
            writer.writeheader()
            writer.writerows(won)
        EXAMPLES.write_text(examples(records) + "\n")
        PLOT.write_text(plot())
        print(f"\nWrote {CSV}, {WINS}, {EXAMPLES} and {PLOT}")


if __name__ == "__main__":
    main()
