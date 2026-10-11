"""Stage 6 tuning: choose the reranker and its candidate depth on questions that are not
the ones the results are reported on.

    python scripts/stage06_tune.py                      # every model already downloaded
    python scripts/stage06_tune.py --models minilm      # one model
    python scripts/stage06_tune.py --download           # also fetch missing weights
    python scripts/stage06_tune.py --write              # also write results/stage06_tuning.csv

As in Stage 5, the choice is made on `questions_pool.jsonl` (at most 25 questions per
evidence cluster, seed 0) and the development questions are scored with the same sweep
only to show how sensitive the choice is. The pool shares articles and gold sentences
with development, so this guards against fitting the wording of the reported questions
and not against fitting their evidence.

Every row reranks the list of configs/stage05_retrieval/hybrid-rrf.yaml. Each model
scores each question's top 100 candidates once; the scores go in the cache named in the
model's config, and depths 20 and 50 are the first 20 and 50 of the same scores. Nothing
here is paid for: the models run on this machine and every question vector is cached.

The listwise LLM reranker is not tuned here. It costs money per question and depth, so
it is run once on the development questions with `rag eval` and reported as a row.
"""

import argparse
import csv
import statistics
import time
from dataclasses import replace
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import try_to_load_from_cache

from ragbasics.config import EvalConfig, PipelineConfig, load_config
from ragbasics.eval import retrieval_metrics as rm
from ragbasics.eval.dataset import read_questions
from ragbasics.eval.runner import count_relevant, score_retrieval
from ragbasics.pipeline import Pipeline
from ragbasics.registry import build
from ragbasics.rerank.base import passage_text, reorder
from ragbasics.types import Candidate, Question
from stage05_tune import CLUSTER_CAP, SEED, MemoEncoding, capped

FIRST_STAGE = Path("configs/stage05_retrieval/hybrid-rrf.yaml")
CONFIGS = Path("configs/stage06_rerank")
EVAL_CONFIG = Path("configs/eval.yaml")
SPLITS = {
    "pool": Path("data/processed/questions_pool.jsonl"),
    "dev": Path("data/processed/questions_dev.jsonl"),
}
CSV = Path("results/stage06_tuning.csv")

MODELS = ["minilm", "bge-v2-m3", "qwen3"]
DEPTHS = [20, 50, 100]
MAX_DEPTH = max(DEPTHS)
METRICS = ["recall@2000tok", "ndcg@10", "full_support@10", "mrr@10", "recall@10"]
DECIDES = "recall@2000tok"
FIELDS = ["split", "model", "depth", "questions", *METRICS, "full_support@2000tok", "recall@50"]
PROGRESS_EVERY = 800  # pairs


def rerank_to(candidates: list[Candidate], scores: list[float], depth: int) -> list[Candidate]:
    """The first-stage list with its top `depth` reordered by `scores`, as the pipeline
    does it: what lies below the depth keeps its order."""
    head = reorder(candidates[:depth], scores[:depth], "rerank")
    tail = [replace(c, rank=depth + i) for i, c in enumerate(candidates[depth:], start=1)]
    return head + tail


class Sweep:
    def __init__(self, pipeline: Pipeline, eval_cfg: EvalConfig, questions: dict[str, list]):
        self.pipeline = pipeline
        self.eval_cfg = eval_cfg
        self.questions = questions
        self.encoding = MemoEncoding()
        everyone = [q for split in questions.values() for q in split]
        self.total_relevant = count_relevant(pipeline.store.chunks, everyone)
        # Each question's hybrid list, once. Every model and depth reranks it.
        self.first: dict[str, list[Candidate]] = {
            q.question_id: pipeline.first_stage(q.text, MAX_DEPTH).candidates for q in everyone
        }
        self.rows: list[dict] = []

    def score(self, split: str, ranked: dict[str, list[Candidate]]) -> dict[str, float]:
        """Mean scores over the split's questions for one ranked list per question."""
        per_question: dict[str, list[float]] = {}
        for question in self.questions[split]:
            chunks = [c.chunk for c in ranked[question.question_id][: self.eval_cfg.depth]]
            scores = score_retrieval(
                question, chunks, self.total_relevant[question.question_id],
                self.eval_cfg, self.encoding,
            )
            scores["full_support@2000tok"] = float(scores["recall@2000tok"] == 1.0)
            scores["recall@50"] = rm.evidence_recall(chunks, question.evidence, self.eval_cfg.depth)
            for name, value in scores.items():
                per_question.setdefault(name, []).append(value)
        means = {name: statistics.fmean(values) for name, values in per_question.items()}
        return {"questions": len(self.questions[split])} | {
            name: means[name] for name in FIELDS if name in means
        }

    def record(self, split: str, model: str, depth: int, ranked: dict) -> dict:
        row = {"split": split, "model": model, "depth": depth, **self.score(split, ranked)}
        self.rows.append(row)
        return row

    def pair_scores(self, reranker, split: str) -> dict[str, list[float]]:
        """The reranker's score for each of every question's top candidates."""
        questions: list[Question] = self.questions[split]
        pairs = [
            (q.text, passage_text(c.chunk, reranker.header))
            for q in questions
            for c in self.first[q.question_id]
        ]
        flat: list[float] = []
        started = time.perf_counter()
        for start in range(0, len(pairs), PROGRESS_EVERY):
            scores, _ = reranker.scores(pairs[start : start + PROGRESS_EVERY])
            flat += scores
            print(f"\r  {split}: {len(flat):,} of {len(pairs):,} pairs, "
                  f"{time.perf_counter() - started:.0f} s", end="", flush=True)
        print()
        out, at = {}, 0
        for q in questions:
            n = len(self.first[q.question_id])
            out[q.question_id] = flat[at : at + n]
            at += n
        return out


def line(row: dict) -> str:
    return "  ".join(
        [f"{row['model']:>10} {row['depth']:>4}"]
        + [f"{100 * row[name]:5.1f}" for name in (*METRICS, "full_support@2000tok", "recall@50")]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--splits", nargs="+", choices=list(SPLITS), default=list(SPLITS),
                        help="score only these; the choice needs the pool")
    parser.add_argument("--download", action="store_true", help="fetch weights not on disk")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    load_dotenv()

    eval_cfg = load_config(EVAL_CONFIG, EvalConfig)
    pipeline = Pipeline(load_config(FIRST_STAGE, PipelineConfig))
    if not pipeline.load():
        raise SystemExit(f"No index in {pipeline.index_dir}.")
    with_evidence = {
        split: [q for q in read_questions(path) if q.evidence] for split, path in SPLITS.items()
    }
    questions = {
        "pool": capped(with_evidence["pool"], CLUSTER_CAP, SEED),
        "dev": with_evidence["dev"],
    }
    questions = {split: questions[split] for split in args.splits}
    texts = [q.text for split in questions.values() for q in split]
    if pipeline.embedder.uncached(texts, kind="query"):
        raise SystemExit("Some question vectors are not cached. Run scripts/stage05_tune.py "
                         "first; this script spends nothing.")
    print(", ".join(f"{len(qs)} {split} questions" for split, qs in questions.items())
          + ". Settings are chosen on the pool. Cost: $0.")

    sweep = Sweep(pipeline, eval_cfg, questions)
    header = "  ".join([f"{'model':>10} {'depth':>4}", "rec2k", " nDCG", "fs@10", "  MRR",
                        "rec10", "fs2k ", "rec50"])
    for split in questions:
        sweep.record(split, "none", 0, sweep.first)

    for key in args.models:
        cfg = load_config(CONFIGS / f"{key}-{MAX_DEPTH}.yaml", PipelineConfig).reranker
        on_disk = isinstance(
            try_to_load_from_cache(cfg.params["model"], "config.json",
                                   revision=cfg.params["revision"]), str)
        if not on_disk and not args.download:
            print(f"\n{key}: {cfg.params['model']} is not downloaded; skipped "
                  "(--download fetches it).")
            continue
        reranker = build("reranker", cfg.name, **cfg.params)
        print(f"\n{key}: {cfg.params['model']}")
        for split in questions:
            scores = sweep.pair_scores(reranker, split)
            for depth in DEPTHS:
                ranked = {
                    qid: rerank_to(sweep.first[qid], scores[qid], depth) for qid in scores
                }
                sweep.record(split, key, depth, ranked)
        if reranker.truncated_pairs:
            print(f"  {reranker.truncated_pairs} pairs were longer than "
                  f"{reranker.scorer.max_length} tokens and were cut.")
        else:
            print(f"  No newly scored pair was longer than {reranker.scorer.max_length} tokens.")

    for split in questions:
        print(f"\n{split} ({len(questions[split])} questions)\n{header}")
        for row in sweep.rows:
            if row["split"] == split:
                print(line(row))
    reranked = [r for r in sweep.rows if r["split"] == "pool" and r["model"] != "none"]
    if reranked:
        best = max(reranked, key=lambda r: r[DECIDES])
        print(f"\nChosen on the pool by {DECIDES}: {best['model']} at depth {best['depth']}.")

    if args.write:
        with open(CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS, restval="")
            writer.writeheader()
            for row in sweep.rows:
                writer.writerow({k: round(v, 4) if isinstance(v, float) else v
                                 for k, v in row.items()})
        print(f"\nWrote {CSV} ({len(sweep.rows)} rows)")


if __name__ == "__main__":
    main()
