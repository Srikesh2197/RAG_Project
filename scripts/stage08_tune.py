"""Stage 8 tuning: which chunks reach the prompt, measured on what the prompt holds.

    python scripts/stage08_tune.py --estimate     # reranker pairs not cached yet, and stop
    python scripts/stage08_tune.py                # the sweep on pool and dev; write the CSV

No model is called and nothing is paid for: the question vectors and the Qwen3-Reranker
scores are cached from Stages 6 and 7 (`--estimate` checks). For each question the
pipeline retrieves and reranks as `rerank-qwen3-100` does, then every assembly setting
is applied to the same ranked list:

    plain        the top k in rank order (today's prompt)
    dedupe       repeated text dropped, the slot refilled from below
    mmr-L-P      maximal marginal relevance with lambda L over the top P
    dedupe+mmr   both

and scored on the assembled context: the share of the gold sentences in it
(context recall), whether all of them are (context full support), and its size. Merging
neighbours and headers change no chunk, only how many passages and tokens there are;
those are reported for the plain setting.

As in Stages 5–7, settings are chosen on `questions_pool.jsonl` (at most 25 questions per
evidence cluster, seed 0); the development questions are scored with the same sweep and
reported once. Differences are paired against `plain` at the same k, with 95% intervals
from redrawing evidence clusters.
"""

import argparse
import csv
from pathlib import Path

from dotenv import load_dotenv

from ragbasics.config import EvalConfig, PipelineConfig, load_config
from ragbasics.context.assemble import Assembler
from ragbasics.eval import retrieval_metrics as rm
from ragbasics.eval.dataset import read_questions
from ragbasics.eval.stats import paired_bootstrap
from ragbasics.pipeline import Pipeline
from ragbasics.rerank.base import passage_text
from ragbasics.rerank.cache import pair_key
from ragbasics.types import Question, Trace
from stage05_tune import CLUSTER_CAP, SEED, capped

CONFIG = Path("configs/stage06_rerank/qwen3-100.yaml")
EVAL_CONFIG = Path("configs/eval.yaml")
SPLITS = {
    "pool": Path("data/processed/questions_pool.jsonl"),
    "dev": Path("data/processed/questions_dev.jsonl"),
}
CSV = Path("results/stage08_tuning.csv")

KS = [3, 5, 10, 20]
LAMBDAS = [0.5, 0.7, 0.9]
POOLS = [20, 50]
METRICS = ["context_recall", "context_full_support"]
FIELDS = ["split", "k", "setting", "questions", "context_recall", "context_full_support",
          "full_support_vs_plain", "full_support_vs_plain_low", "full_support_vs_plain_high",
          "recall_vs_plain", "recall_vs_plain_low", "recall_vs_plain_high",
          "chunks_swapped", "duplicates", "passages_merged", "passages_header",
          "tokens", "tokens_header", "tokens_merged_header"]


def settings(k: int) -> dict[str, Assembler]:
    out = {"plain": Assembler(top_k=k), "dedupe": Assembler(top_k=k, pool=50, dedupe=True)}
    for pool in POOLS:
        if pool <= k:
            continue
        for lam in LAMBDAS:
            out[f"mmr-{lam}-{pool}"] = Assembler(top_k=k, pool=pool, mmr=lam)
            out[f"dedupe+mmr-{lam}-{pool}"] = Assembler(top_k=k, pool=pool, mmr=lam,
                                                        dedupe=True)
    return out


def ranked(pipeline: Pipeline, questions: list[Question], depth: int) -> dict[str, Trace]:
    return {q.question_id: pipeline.retrieve(q.text, depth) for q in questions}


def uncached_pairs(pipeline: Pipeline, questions: list[Question], depth: int) -> int:
    reranker = pipeline.reranker
    keys = []
    for q in questions:
        trace = pipeline.first_stage(q.text, depth)
        keys += [pair_key(reranker.scorer.identity, q.text, passage_text(c.chunk, reranker.header))
                 for c in trace.candidates[: reranker.depth]]
    known = reranker.cache.get_many(keys)
    return len(set(keys) - set(known))


def score(q: Question, chunks) -> dict[str, float]:
    return {
        "context_recall": rm.evidence_recall(chunks, q.evidence, len(chunks)),
        "context_full_support": rm.full_support(chunks, q.evidence, len(chunks)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--estimate", action="store_true", help="count uncached pairs and stop")
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=list(SPLITS))
    args = parser.parse_args()
    load_dotenv()

    eval_cfg = load_config(EVAL_CONFIG, EvalConfig)
    pipeline = Pipeline(load_config(CONFIG, PipelineConfig))
    pipeline.load()
    vectors = pipeline.store.vectors_for
    questions = {}
    for split in args.splits:
        with_evidence = [q for q in read_questions(SPLITS[split]) if q.evidence]
        questions[split] = (capped(with_evidence, CLUSTER_CAP, SEED) if split == "pool"
                            else with_evidence)
        print(f"{split}: {len(questions[split])} questions with gold evidence")

    if args.estimate:
        for split, qs in questions.items():
            missing = uncached_pairs(pipeline, qs, eval_cfg.depth)
            print(f"{split}: {missing} reranker pairs not in the cache"
                  + (" (the sweep needs no model)" if not missing else ""))
        return

    rows = []
    for split, qs in questions.items():
        traces = ranked(pipeline, qs, eval_cfg.depth)
        clusters = [q.cluster for q in qs]
        for k in KS:
            per: dict[str, list[dict[str, float]]] = {}
            notes: dict[str, list[dict]] = {}
            for name, assembler in settings(k).items():
                assemblies = [assembler.assemble(traces[q.question_id].candidates, vectors)
                              for q in qs]
                per[name] = [score(q, a.chunks) for q, a in zip(qs, assemblies, strict=True)]
                notes[name] = [a.notes for a in assemblies]
            # What merging and headers do to the plain setting's size.
            plain_lists = [traces[q.question_id].candidates for q in qs]
            sizes = {
                label: [Assembler(top_k=k, **params).assemble(c).notes for c in plain_lists]
                for label, params in {
                    "header": {"header": True}, "merged_header": {"merge": True, "header": True},
                }.items()
            }
            for name, scores in per.items():
                row = {"split": split, "k": k, "setting": name, "questions": len(qs)}
                for metric in METRICS:
                    row[metric] = round(100 * sum(s[metric] for s in scores) / len(qs), 2)
                for metric, label in (("context_full_support", "full_support"),
                                      ("context_recall", "recall")):
                    if name == "plain":
                        continue
                    diff = paired_bootstrap([s[metric] for s in scores],
                                            [s[metric] for s in per["plain"]], clusters,
                                            eval_cfg.bootstrap.samples)
                    row[f"{label}_vs_plain"] = round(100 * diff.mean, 2)
                    row[f"{label}_vs_plain_low"] = round(100 * diff.low, 2)
                    row[f"{label}_vs_plain_high"] = round(100 * diff.high, 2)
                n = len(qs)
                row["chunks_swapped"] = round(
                    sum(x.get("mmr_swapped", 0) for x in notes[name]) / n, 2)
                row["duplicates"] = round(sum(x.get("duplicates", 0) for x in notes[name]) / n, 2)
                row["tokens"] = round(sum(x["tokens"] for x in notes[name]) / n, 1)
                if name == "plain":
                    row["tokens_header"] = round(sum(x["tokens"] for x in sizes["header"]) / n, 1)
                    row["tokens_merged_header"] = round(
                        sum(x["tokens"] for x in sizes["merged_header"]) / n, 1)
                    row["passages_header"] = k
                    row["passages_merged"] = round(
                        sum(x["passages"] for x in sizes["merged_header"]) / n, 2)
                rows.append(row)

    CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {CSV}")

    for split in questions:
        print(f"\n{split}: context full support (recall), difference from plain at the same k")
        for k in KS:
            print(f"  k={k}")
            for row in [r for r in rows if r["split"] == split and r["k"] == k]:
                diff = ("" if row["setting"] == "plain" else
                        f"  {row['full_support_vs_plain']:+.1f} "
                        f"({row['full_support_vs_plain_low']:+.1f} to "
                        f"{row['full_support_vs_plain_high']:+.1f})")
                extra = (f"  tokens {row['tokens']:.0f}, with header {row['tokens_header']:.0f}, "
                         f"merged {row['tokens_merged_header']:.0f} in "
                         f"{row['passages_merged']:.1f} passages"
                         if row["setting"] == "plain" else
                         f"  swapped {row['chunks_swapped']}, dropped {row['duplicates']}")
                print(f"    {row['setting']:<22} {row['context_full_support']:5.1f} "
                      f"({row['context_recall']:5.1f}){diff}{extra}")


if __name__ == "__main__":
    main()
