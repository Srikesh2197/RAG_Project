"""Stage 6 diagnostic: why does a cross-encoder push down chunks that hold evidence?

    python scripts/stage06_clauses.py                    # MiniLM, development questions
    python scripts/stage06_clauses.py --model bge-v2-m3  # another downloaded model
    python scripts/stage06_clauses.py --write            # also write results/stage06_clauses.csv

The explanation being tested: these questions ask about several facts at once, each
gold chunk supports one of them, and a cross-encoder trained on single-fact queries
scores a chunk against the whole question, so a chunk that supports one fact in four
looks like a poor match.

The test: cut each question into clauses with a fixed rule (no model), and score every
candidate against each clause as well as against the whole question. If the explanation
holds, a gold chunk should stand out from the other candidates more clearly under its
best clause than under the whole question.

Three ways of ordering each question's top candidates are compared:

    full         the reranker's score against the whole question (the Stage 6 rows)
    best clause  the highest of the chunk's scores against the clauses
    round robin  each clause ranks the candidates; take every clause's first choice,
                 then every clause's second, and so on

"Best clause" compares scores across clauses, and one clause can score everything
higher than another. Round robin never compares across clauses.

What this does not separate: a clause is also a shorter query, closer to what the model
was trained on. A gain here shows that splitting helps, not which of the two reasons.

Nothing is chosen from this; it explains a result. The clause rule is crude on purpose.
Stage 7 builds decomposition properly, with a model. Cost: $0.
"""

import argparse
import csv
import re
import statistics
from pathlib import Path

from dotenv import load_dotenv

from ragbasics.config import EvalConfig, PipelineConfig, load_config
from ragbasics.eval import retrieval_metrics as rm
from ragbasics.eval.dataset import read_questions
from ragbasics.eval.runner import count_relevant, score_retrieval
from ragbasics.eval.stats import paired_bootstrap
from ragbasics.pipeline import Pipeline
from ragbasics.registry import build
from ragbasics.rerank.base import passage_text
from ragbasics.types import Candidate, Question
from stage05_tune import MemoEncoding
from stage06_tune import CONFIGS, EVAL_CONFIG, FIRST_STAGE, MAX_DEPTH, MODELS, SPLITS

CSV = Path("results/stage06_clauses.csv")
DEPTHS = [20, 50, 100]
METRICS = ["recall@2000tok", "ndcg@10", "full_support@10", "mrr@10"]
TYPES = {"comparison_query": "comparison", "inference_query": "inference",
         "temporal_query": "temporal"}
MIN_WORDS = 5  # a piece shorter than this joins the piece before it
BOUNDARY = re.compile(r",\s+|\s+(?:while|whereas|and|but|before|after)\s+")


def split_clauses(question: str) -> list[str]:
    """Cut a question at commas and at a few joining words.

    A piece under five words (", 2023," or "and Fortune") is glued back onto the piece
    before it, so dates and short lists stay whole. A question with no boundary is one
    clause.
    """
    spans: list[list[int]] = []  # [start, end) of each clause in the question
    start = 0
    for match in [*BOUNDARY.finditer(question), None]:
        end = match.start() if match else len(question)
        if spans and len(question[start:end].split()) < MIN_WORDS:
            spans[-1][1] = end
        elif question[start:end].strip():
            spans.append([start, end])
        start = match.end() if match else end
    # A short first piece joins the one after it.
    if len(spans) > 1 and len(question[spans[0][0] : spans[0][1]].split()) < MIN_WORDS:
        spans[1][0] = spans[0][0]
        spans = spans[1:]
    return [question[a:b].strip() for a, b in spans]


def by_score(scores: list[float]) -> list[int]:
    """Candidate positions, best score first; ties keep first-stage order."""
    return sorted(range(len(scores)), key=lambda i: (-scores[i], i))


def round_robin(per_clause: list[list[float]]) -> list[int]:
    """Every clause's first choice, then every clause's second, and so on."""
    orders = [by_score(scores) for scores in per_clause]
    seen: set[int] = set()
    out: list[int] = []
    for position in range(len(orders[0])):
        for order in orders:
            if order[position] not in seen:
                seen.add(order[position])
                out.append(order[position])
    return out


def separation(order: list[int], relevant: list[bool]) -> float | None:
    """The chance that a gold chunk is ranked above a chunk that is not gold: 1 when
    every gold chunk is on top, 0.5 when the order says nothing. None if the list holds
    no gold chunk or nothing else."""
    place = {i: rank for rank, i in enumerate(order)}
    gold = [place[i] for i, hit in enumerate(relevant) if hit]
    rest = [place[i] for i, hit in enumerate(relevant) if not hit]
    if not gold or not rest:
        return None
    return statistics.fmean(float(g < r) for g in gold for r in rest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, default="minilm")
    parser.add_argument("--split", choices=list(SPLITS), default="dev")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    load_dotenv()

    eval_cfg = load_config(EVAL_CONFIG, EvalConfig)
    pipeline = Pipeline(load_config(FIRST_STAGE, PipelineConfig))
    if not pipeline.load():
        raise SystemExit(f"No index in {pipeline.index_dir}.")
    spec = load_config(CONFIGS / f"{args.model}-{MAX_DEPTH}.yaml", PipelineConfig).reranker
    reranker = build("reranker", spec.name, **spec.params)
    questions: list[Question] = [q for q in read_questions(SPLITS[args.split]) if q.evidence]
    encoding = MemoEncoding()
    total_relevant = count_relevant(pipeline.store.chunks, questions)

    clauses = {q.question_id: split_clauses(q.text) for q in questions}
    counts = [len(c) for c in clauses.values()]
    print(f"{len(questions)} {args.split} questions, {args.model}. Clauses per question: "
          f"median {statistics.median(counts)}, mean {statistics.fmean(counts):.1f}, "
          f"max {max(counts)}; {sum(n == 1 for n in counts)} questions are one clause.")
    example = max(questions[:40], key=lambda q: len(clauses[q.question_id]))
    print(f"Example: {example.text}")
    for clause in clauses[example.question_id]:
        print(f"  - {clause}")

    # --- Score every candidate against the whole question and against each clause ----------
    first: dict[str, list[Candidate]] = {}
    orders: dict[str, dict[str, list[int]]] = {"first stage": {}, "full": {}, "best clause": {},
                                               "round robin": {}}
    relevant: dict[str, list[bool]] = {}
    raw: dict[str, tuple[list[float], list[list[float]]]] = {}  # full, and one list per clause
    gold_gain: list[float] = []  # best-clause score minus full score, gold chunks
    rest_gain: list[float] = []  # the same for the other candidates
    for n, q in enumerate(questions, start=1):
        found = pipeline.first_stage(q.text, MAX_DEPTH).candidates
        passages = [passage_text(c.chunk, reranker.header) for c in found]
        full, _ = reranker.scores([(q.text, p) for p in passages])
        per_clause = [
            reranker.scores([(clause, p) for p in passages])[0]
            for clause in clauses[q.question_id]
        ]
        best = [max(scores[i] for scores in per_clause) for i in range(len(found))]
        first[q.question_id] = found
        raw[q.question_id] = (full, per_clause)
        relevant[q.question_id] = rm.relevance([c.chunk for c in found], q.evidence)
        orders["first stage"][q.question_id] = list(range(len(found)))
        orders["full"][q.question_id] = by_score(full)
        orders["best clause"][q.question_id] = by_score(best)
        orders["round robin"][q.question_id] = round_robin(per_clause)
        for i, hit in enumerate(relevant[q.question_id]):
            (gold_gain if hit else rest_gain).append(best[i] - full[i])
        print(f"\r  scored {n} of {len(questions)} questions", end="", flush=True)
    print()

    # --- 1. Does a gold chunk stand out more under its best clause? ------------------------
    print(f"\nChance that a gold chunk is ranked above a non-gold one, among each question's "
          f"top {MAX_DEPTH} hybrid candidates (0.5 says nothing, 1 is perfect):")
    print(f"{'order':>12}  " + "  ".join(f"{label:>10}" for label in ("all", *TYPES.values())))
    rows: list[dict] = []
    for name, order in orders.items():
        cells = []
        for kind in (None, *TYPES):
            values = [
                s for q in questions
                if (kind is None or q.question_type == kind)
                and (s := separation(order[q.question_id], relevant[q.question_id])) is not None
            ]
            cells.append(statistics.fmean(values))
            rows.append({"table": "separation", "order": name, "depth": MAX_DEPTH,
                         "question_type": TYPES.get(kind, "all"), "value": cells[-1],
                         "questions": len(values)})
        print(f"{name:>12}  " + "  ".join(f"{cell:10.3f}" for cell in cells))
    print(f"\nScore under the best clause minus score under the whole question, median: "
          f"gold chunks {statistics.median(gold_gain):+.2f} ({len(gold_gain)}), other "
          f"candidates {statistics.median(rest_gain):+.2f} ({len(rest_gain)}).")

    # --- 2. What the three orders do to the retrieval metrics ------------------------------
    def scores_for(name: str, depth: int) -> dict[str, dict[str, float]]:
        """Retrieval metrics with the top `depth` reordered and the rest left alone, as
        the pipeline does it."""
        out = {}
        for q in questions:
            found = first[q.question_id]
            full, per_clause = raw[q.question_id]
            if name == "first stage":
                head = list(range(depth))
            elif name == "full":
                head = by_score(full[:depth])
            elif name == "best clause":
                head = by_score([max(s[i] for s in per_clause) for i in range(depth)])
            else:
                head = round_robin([s[:depth] for s in per_clause])
            order = head + list(range(depth, len(found)))
            chunks = [found[i].chunk for i in order][: eval_cfg.depth]
            out[q.question_id] = score_retrieval(
                q, chunks, total_relevant[q.question_id], eval_cfg, encoding
            )
        return out

    clusters = [q.cluster for q in questions]
    baseline = scores_for("first stage", 0)

    def mean(scored: dict, metric: str) -> float:
        return statistics.fmean(scored[q.question_id][metric] for q in questions)

    def versus(a: dict, b: dict, metric: str) -> str:
        d = paired_bootstrap(
            [a[q.question_id][metric] for q in questions],
            [b[q.question_id][metric] for q in questions], clusters,
        )
        return f"{100 * d.mean:+.1f} ({100 * d.low:+.1f} to {100 * d.high:+.1f})"

    print(f"\n{'order':>12} {'depth':>5}  " + "  ".join(f"{m:>15}" for m in METRICS)
          + "   recall@2000tok vs hybrid / vs full question")
    print(f"{'first stage':>12} {'':>5}  "
          + "  ".join(f"{100 * mean(baseline, m):15.1f}" for m in METRICS))
    for depth in DEPTHS:
        full_scored = scores_for("full", depth)
        for name in ("full", "best clause", "round robin"):
            scored = full_scored if name == "full" else scores_for(name, depth)
            against_full = "" if name == "full" else versus(scored, full_scored, METRICS[0])
            print(f"{name:>12} {depth:>5}  "
                  + "  ".join(f"{100 * mean(scored, m):15.1f}" for m in METRICS)
                  + f"   {versus(scored, baseline, METRICS[0])} / {against_full}")
            for metric in METRICS:
                rows.append({"table": "metrics", "order": name, "depth": depth,
                             "question_type": "all", "metric": metric,
                             "value": mean(scored, metric), "questions": len(questions)})

    if args.write:
        fields = ["table", "order", "depth", "question_type", "metric", "value", "questions"]
        with open(CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, restval="")
            writer.writeheader()
            for row in rows:
                rounded = {k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}
                writer.writerow(rounded | {"table": f"{args.model}:{row['table']}"})
        print(f"\nWrote {CSV} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
