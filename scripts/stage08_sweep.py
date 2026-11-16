"""Stage 8 tables: what reaches the prompt, and what that does to the answers.

    python scripts/stage08_sweep.py            # print the tables (runs that exist so far)
    python scripts/stage08_sweep.py --write    # also results/stage08_context.csv and the plot

Reads finished runs in runs/ and calls no model. Four tables:

    free     full support of the top k, k = 3, 5, 10, 20, for the Stage 6 and 7 rankings
             (Qwen3 at depth 100, the listwise LLM reranker, Qwen3 scoring decomposed
             questions), from their retrieval-only runs, on 441 development questions
    header   today's prompt against the same prompt with a title, source and date line on
             each passage, and the gold evidence with and without that line
    size     answers against 3, 5, 10 and 20 chunks
    setting  orderings, merging, diversity and the LLM reranker at the chosen size

Answers are the 150-question subset. Differences are paired, in percentage points, with
95% intervals from redrawing evidence clusters.
"""

import argparse
import csv
from pathlib import Path

from ragbasics.eval.report import compare, load_run

RUNS = Path("runs")
CSV = Path("results/stage08_context.csv")
PLOT = Path("results/stage08_context_size.svg")

KS = [3, 5, 10, 20]
RANKINGS = {
    "rerank-qwen3-100": "Qwen3-Reranker, depth 100 (best so far)",
    "rerank-llm-luna-50": "listwise LLM reranker gpt-6-luna, depth 50 (Stage 6)",
    "query-decompose-rerank-split-50": "Qwen3 scoring decomposed questions, depth 50 (Stage 7)",
}
REFERENCE = "ctx-k5"
TYPES = {"comparison_query": "comparison", "inference_query": "inference",
         "temporal_query": "temporal", "null_query": "null"}
ANSWER_METRICS = ["correct", "correct_answerable", "false_abstain", "faithful",
                  "context_full_support", "context_tokens"]


def load(run_id: str) -> dict | None:
    path = RUNS / run_id
    return load_run(path) if (path / "metrics.json").exists() else None


def pct(value: float | None, digits: int = 1) -> str:
    return "" if value is None else f"{100 * value:.{digits}f}"


def signed(interval) -> str:
    if interval is None or interval.n == 0:
        return ""
    return (f"{100 * interval.mean:+.1f} ({100 * interval.low:+.0f} to "
            f"{100 * interval.high:+.0f})")


def mean(run: dict, metric: str, subset: bool = True) -> float | None:
    values = [row["scores"][metric] for row in run["rows"]
              if metric in row["scores"] and (row["answer_subset"] or not subset)]
    return sum(values) / len(values) if values else None


def by_type(run: dict, qtype: str) -> float | None:
    m = run["by_type"].get(qtype, {}).get("correct")
    return m["mean"] if m else None


# --- Free: full support of the top k by ranking --------------------------------------------


def top_k_scores(run: dict, k: int) -> dict:
    """A retrieval-only run with gold ranks, re-scored as full support of its top k."""
    rows = []
    for row in run["rows"]:
        ranks = row.get("gold_ranks")
        if ranks is None:
            continue
        ranks = [r if r is not None else 10**6 for r in ranks]
        rows.append({"question_id": row["question_id"], "cluster": row["cluster"],
                     "scores": {"fs": float(max(ranks) <= k),
                                "recall": sum(r <= k for r in ranks) / len(ranks)}})
    return {"rows": rows}


def free_table() -> list[str]:
    runs = {rid: load(rid) for rid in RANKINGS}
    if any(r is None for r in runs.values()):
        return []
    lines = ["Full support of the top k (441 development questions; free, from retrieval runs)",
             "", "| ranking | " + " | ".join(f"k={k}" for k in KS) + " |",
             "|---|" + "---|" * len(KS)]
    best = runs["rerank-qwen3-100"]
    for rid, label in RANKINGS.items():
        cells = []
        for k in KS:
            mine, ref = top_k_scores(runs[rid], k), top_k_scores(best, k)
            value = sum(r["scores"]["fs"] for r in mine["rows"]) / len(mine["rows"])
            cell = pct(value)
            if rid != "rerank-qwen3-100":
                cell += f", {signed(compare(mine, ref, 'fs'))}"
            cells.append(cell)
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return lines


# --- Answer tables -------------------------------------------------------------------------


def answer_row(run_id: str, label: str, reference: dict | None) -> dict | None:
    run = load(run_id)
    if run is None:
        return None
    row = {"run": run_id, "label": label}
    for metric in ANSWER_METRICS:
        row[metric] = mean(run, metric)
    for qtype, name in TYPES.items():
        row[name] = by_type(run, qtype)
    tokens = [r["usage"]["input_tokens"] for r in run["rows"] if r.get("usage")]
    row["prompt_tokens"] = sum(tokens) / len(tokens) if tokens else None
    row["cost"] = run["cost_usd"]["list_price"]
    if reference is not None and run["run_id"] != reference["run_id"]:
        row["vs"] = compare(run, reference, "correct")
        row["vs_temporal"] = compare(
            {"rows": [r for r in run["rows"] if r["question_type"] == "temporal_query"]},
            reference, "correct")
    row["_run"] = run
    return row


def answer_table(title: str, specs: list[tuple[str, str]], reference_id: str) -> list[dict]:
    reference = load(reference_id)
    rows = [r for rid, label in specs if (r := answer_row(rid, label, reference))]
    if not rows:
        return []
    print(f"\n{title} (150 questions; differences against {reference_id})\n")
    print("| run | correct | vs reference | temporal | temporal vs reference | comparison "
          "| inference | false abstentions | faithful* | full support in prompt "
          "| prompt tokens | cost |")
    print("|---|" + "---|" * 11)
    for r in rows:
        print(f"| {r['label']} | {pct(r['correct'])} | {signed(r.get('vs'))} "
              f"| {pct(r['temporal'])} | {signed(r.get('vs_temporal'))} "
              f"| {pct(r['comparison'])} | {pct(r['inference'])} | {pct(r['false_abstain'])} "
              f"| {pct(r['faithful'])} | {pct(r['context_full_support'])} "
              f"| {r['prompt_tokens'] or 0:,.0f} | ${r['cost']:.2f} |")
    return rows


def plot(rows: list[dict]) -> str:
    """Correct and full support in the prompt against chunks in the prompt, as an SVG."""
    points = [(k, r) for k in KS for r in rows if r["run"] == size_run(k)]
    width, height, left, top, right, bottom = 640, 400, 60, 40, 600, 330

    def x(k: int) -> float:
        return left + (right - left) * KS.index(k) / (len(KS) - 1)

    def y(share: float) -> float:
        return bottom - (bottom - top) * share  # 0% to 100%

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        'font-family="sans-serif" font-size="12">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        f'<text x="{left}" y="20" font-size="14">Answers against chunks in the prompt '
        "(150 development questions)</text>",
    ]
    for tick in range(0, 101, 20):
        ty = y(tick / 100)
        out.append(f'<line x1="{left}" y1="{ty}" x2="{right}" y2="{ty}" stroke="#ddd"/>')
        out.append(f'<text x="{left - 8}" y="{ty + 4}" text-anchor="end">{tick}%</text>')
    series = [("correct", "#1f5fa8", "", "answer correct"),
              ("context_full_support", "#c0632b", "6 4", "all gold sentences in the prompt"),
              ("faithful", "#6a8f3a", "2 4", "faithful (judge unvalidated)")]
    for i, (metric, colour, dash, label) in enumerate(series):
        pts = [(k, r[metric]) for k, r in points if r[metric] is not None]
        if not pts:
            continue
        path = " ".join(f"{x(k):.1f},{y(v):.1f}" for k, v in pts)
        out.append(f'<polyline points="{path}" fill="none" stroke="{colour}" stroke-width="2"'
                   + (f' stroke-dasharray="{dash}"' if dash else "") + "/>")
        for k, v in pts:
            out.append(f'<circle cx="{x(k):.1f}" cy="{y(v):.1f}" r="4" fill="{colour}"/>')
            out.append(f'<text x="{x(k) + 7:.1f}" y="{y(v) - 7:.1f}" fill="{colour}">'
                       f"{100 * v:.0f}</text>")
        ly = y(0.16) + 16 * i  # the empty band under the curves
        out.append(f'<line x1="{right - 230}" y1="{ly}" x2="{right - 205}" y2="{ly}" '
                   f'stroke="{colour}" stroke-width="2"' + (f' stroke-dasharray="{dash}"'
                                                            if dash else "") + "/>")
        out.append(f'<text x="{right - 198}" y="{ly + 4}">{label}</text>')
    # Today's prompt at 5 chunks, for the size of the header's effect.
    plain = load(REFERENCE)
    if plain is not None:
        v = plain["metrics"]["correct"]["mean"]
        out.append(f'<circle cx="{x(5):.1f}" cy="{y(v):.1f}" r="4" fill="white" '
                   'stroke="#1f5fa8" stroke-width="2"/>')
        out.append(f'<text x="{x(5) + 8:.1f}" y="{y(v) + 4:.1f}" fill="#1f5fa8">{100 * v:.0f}, '
                   "no header</text>")
    # 95% interval on correctness.
    for k, r in points:
        m = r["_run"]["metrics"]["correct"]
        out.append(f'<line x1="{x(k):.1f}" y1="{y(m["low"]):.1f}" x2="{x(k):.1f}" '
                   f'y2="{y(m["high"]):.1f}" stroke="#1f5fa8"/>')
        out.append(f'<text x="{x(k)}" y="{bottom + 18}" text-anchor="middle">{k}</text>')
        out.append(f'<text x="{x(k)}" y="{bottom + 32}" text-anchor="middle" fill="#666">'
                   f'{r["prompt_tokens"] or 0:,.0f} tok</text>')
    out.append(f'<text x="{(left + right) / 2}" y="{bottom + 56}" text-anchor="middle">chunks '
               "in the prompt, each with a title, source and date line; bars are 95% "
               "intervals over evidence clusters</text>")
    out.append("</svg>")
    return "\n".join(out) + "\n"


def size_run(k: int) -> str:
    return f"ctx-k{k}-header"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--size", type=int, default=None,
                        help="the chosen number of chunks, for the settings table")
    args = parser.parse_args()

    print("\n".join(free_table()))
    header = answer_table("Header", [
        ("ctx-k5", "5 chunks, today's prompt"),
        ("ctx-k5-header", "5 chunks, with header"),
    ], REFERENCE)
    header += answer_table("Header on gold evidence", [
        ("baseline-gold", "gold evidence, no header (Stage 2)"),
        ("ctx-k5-header-gold", "gold evidence, with header"),
    ], "baseline-gold")
    size = answer_table("Context size", [(size_run(k), f"{k} chunks") for k in KS]
                        + [("ctx-k5", "5 chunks, no header")], REFERENCE)
    settings = []
    if args.size:
        base = size_run(args.size)
        settings = answer_table(f"Settings at {args.size} chunks", [
            (base, "rank order"),
            (f"{base}-edges", "best at the edges"),
            (f"{base}-chronological", "by article date"),
            (f"{base}-merge", "dedupe and merge neighbours"),
            (f"{base}-mmr", "diversity (MMR)"),
            (f"{base}-luna", "listwise LLM reranker"),
        ], base)
    if args.write:
        rows = header + size + settings
        with open(CSV, "w", newline="") as f:
            fields = ["run", "label", *ANSWER_METRICS, *TYPES.values(), "prompt_tokens", "cost",
                      "vs", "vs_low", "vs_high"]
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            seen = set()
            for r in rows:
                if r["run"] in seen:
                    continue
                seen.add(r["run"])
                out = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()
                       if not k.startswith("_") and k not in ("vs", "vs_temporal")}
                if r.get("vs"):
                    out |= {"vs": round(r["vs"].mean, 4), "vs_low": round(r["vs"].low, 4),
                            "vs_high": round(r["vs"].high, 4)}
                writer.writerow(out)
        PLOT.write_text(plot(size))
        print(f"\nWrote {CSV} and {PLOT}")


if __name__ == "__main__":
    main()
