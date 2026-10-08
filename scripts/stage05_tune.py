"""Stage 5 tuning: choose the BM25 and fusion settings on questions that are not the ones
the results are reported on.

    python scripts/stage05_tune.py --estimate   # print the embedding cost and stop
    python scripts/stage05_tune.py              # print the sweeps and the chosen settings
    python scripts/stage05_tune.py --write      # also write results/stage05_tuning.csv

Settings are chosen on `questions_pool.jsonl`: the development-side questions left over
by the split. They are different questions from the 441 the stage reports on. They are
about the same articles and often the same gold sentences, so this guards against fitting
the wording of the development questions and not against fitting their evidence; the
held-out test in Stage 12 is the check on that.

One cluster (questions about one sentence on Sam Bankman-Fried) supplies 424 of the
pool's 687 questions with evidence. Each cluster is capped at 25 questions, drawn with a
fixed seed, so no single sentence decides the settings.

The same sweeps are also run on the development questions and written to the CSV. Those
rows describe how sensitive each setting is. They are not used to choose anything.

Everything ranks the chunks of data/index/chunk-recursive-128. The only paid step is
embedding the pool questions, about a tenth of a cent, once: they go in the cache.
"""

import argparse
import csv
import random
import statistics
from itertools import product
from pathlib import Path

import tiktoken
from dotenv import load_dotenv

from ragbasics.config import EvalConfig, PipelineConfig, load_config
from ragbasics.costs import append_ledger, embedding_cost
from ragbasics.eval import retrieval_metrics as rm
from ragbasics.eval.dataset import read_questions
from ragbasics.eval.runner import count_relevant, score_retrieval
from ragbasics.pipeline import Pipeline
from ragbasics.retrieval.fusion import reciprocal_rank_fusion, weighted_fusion
from ragbasics.retrieval.sparse import SparseRetriever
from ragbasics.types import Candidate, Question

BEST_CONFIG = Path("configs/stage03_chunking/recursive-128.yaml")
EVAL_CONFIG = Path("configs/eval.yaml")
EMBEDDING_CACHE = Path("data/cache/embeddings.sqlite")
SPLITS = {
    "pool": Path("data/processed/questions_pool.jsonl"),
    "dev": Path("data/processed/questions_dev.jsonl"),
}
LEDGER = Path("runs/cost_ledger.csv")
CSV = Path("results/stage05_tuning.csv")

CLUSTER_CAP = 25
SEED = 0
RECALL, NDCG = "recall@2000tok", "ndcg@10"

# What is swept. Lowercasing and the rule for what a term is are fixed.
TOKENISERS = [
    {"stopwords": stopwords, "stemmer": stemmer}
    for stopwords, stemmer in product([False, True], ["none", "porter"])
]
K1 = [0.4, 0.8, 1.2, 1.6, 2.0]
B = [0.0, 0.25, 0.5, 0.75, 1.0]
BM25_DEFAULT = {"stopwords": False, "stemmer": "none", "k1": 1.2, "b": 0.75}
RRF_K = [1, 5, 10, 20, 40, 60, 100, 200]
WEIGHTS = [round(0.1 * i, 1) for i in range(11)]  # the dense list's share
NORMALISERS = ["minmax", "zscore"]
DEPTH = 200  # chunks from each retriever before fusion
DEPTHS = [50, 100, 200, 500, 1000]
FIELDS = [
    "split", "family", "stopwords", "stemmer", "header", "k1", "b", "method", "rrf_k",
    "weight", "normalise", "depth", "questions", "recall_2000tok", "ndcg@10", "recall@50",
]


class MemoEncoding:
    """tiktoken with the token list of each text remembered: the same chunks are scored
    thousands of times in a sweep."""

    def __init__(self) -> None:
        self._encoding = tiktoken.get_encoding("cl100k_base")
        self._tokens: dict[str, list[int]] = {}

    def encode(self, text: str, disallowed_special: tuple = ()) -> list[int]:
        if text not in self._tokens:
            self._tokens[text] = self._encoding.encode(text, disallowed_special=())
        return self._tokens[text]

    def decode(self, tokens: list[int]) -> str:
        return self._encoding.decode(tokens)


def capped(questions: list[Question], cap: int, seed: int) -> list[Question]:
    """At most `cap` questions from each evidence cluster, in the original order."""
    rng = random.Random(seed)
    by_cluster: dict[str, list[Question]] = {}
    for question in questions:
        by_cluster.setdefault(question.cluster, []).append(question)
    kept = set()
    for members in by_cluster.values():
        chosen = members if len(members) <= cap else rng.sample(members, cap)
        kept |= {question.question_id for question in chosen}
    return [question for question in questions if question.question_id in kept]


class Sweep:
    def __init__(self, pipeline: Pipeline, eval_cfg: EvalConfig, questions: dict[str, list]):
        self.pipeline = pipeline
        self.eval_cfg = eval_cfg
        self.questions = questions
        self.encoding = MemoEncoding()
        everyone = [q for split in questions.values() for q in split]
        self.total_relevant = count_relevant(pipeline.store.chunks, everyone)
        self.dense: dict[str, list[Candidate]] = {}
        self.rows: list[dict] = []

    def search_dense(self, depth: int) -> None:
        """Each question's dense list, once. Every fusion setting reuses it."""
        store = self.pipeline.store
        for split in self.questions.values():
            texts = [q.text for q in split]
            vectors = self.pipeline.embedder.embed(texts, kind="query").vectors
            for question, vector in zip(split, vectors, strict=True):
                self.dense[question.question_id] = store.search(vector, depth)

    def search_sparse(self, retriever: SparseRetriever, split: str, depth: int) -> dict:
        return {
            q.question_id: retriever.retrieve(q.text, depth, self.pipeline).candidates
            for q in self.questions[split]
        }

    def score(self, split: str, ranked: dict[str, list[Candidate]]) -> dict[str, float]:
        """Mean scores over the split's questions for one ranked list per question."""
        recall, ndcg, deep = [], [], []
        for question in self.questions[split]:
            chunks = [c.chunk for c in ranked[question.question_id][: self.eval_cfg.depth]]
            scores = score_retrieval(
                question, chunks, self.total_relevant[question.question_id],
                self.eval_cfg, self.encoding,
            )
            recall.append(scores[RECALL])
            ndcg.append(scores[NDCG])
            deep.append(rm.evidence_recall(chunks, question.evidence, self.eval_cfg.depth))
        return {
            "questions": len(recall),
            "recall_2000tok": statistics.fmean(recall),
            "ndcg@10": statistics.fmean(ndcg),
            "recall@50": statistics.fmean(deep),
        }

    def record(self, split: str, family: str, ranked: dict, **settings) -> dict:
        row = {"split": split, "family": family, **settings, **self.score(split, ranked)}
        self.rows.append(row)
        return row

    def best(self, family: str, split: str = "pool") -> dict:
        """The setting with the highest recall in 2,000 tokens on the tuning questions.
        Equal scores go to the row that came first in the sweep."""
        rows = [r for r in self.rows if r["family"] == family and r["split"] == split]
        return max(rows, key=lambda r: r["recall_2000tok"])


def fuse(dense: list, sparse: list, depth: int, method: str, **settings) -> list[Candidate]:
    lists = {"dense": dense[:depth], "sparse": sparse[:depth]}
    if method == "rrf":
        return reciprocal_rank_fusion(lists, settings["rrf_k"])
    weights = {"dense": settings["weight"], "sparse": 1 - settings["weight"]}
    return weighted_fusion(lists, weights, settings["normalise"])


def pct(row: dict) -> str:
    return (f"recall in 2,000 tok {100 * row['recall_2000tok']:.1f}, nDCG@10 "
            f"{100 * row['ndcg@10']:.1f}, recall@50 {100 * row['recall@50']:.1f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--estimate", action="store_true", help="print the cost and stop")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    load_dotenv()

    cfg = load_config(BEST_CONFIG, PipelineConfig).model_copy(
        update={"embedding_cache": EMBEDDING_CACHE}
    )
    eval_cfg = load_config(EVAL_CONFIG, EvalConfig)
    pipeline = Pipeline(cfg)
    if not pipeline.load():
        raise SystemExit(f"No index in {pipeline.index_dir}.")

    with_evidence = {
        split: [q for q in read_questions(path) if q.evidence] for split, path in SPLITS.items()
    }
    questions = {
        "pool": capped(with_evidence["pool"], CLUSTER_CAP, SEED),
        "dev": with_evidence["dev"],
    }
    clusters = len({q.cluster for q in questions["pool"]})
    print(f"Tuning on {len(questions['pool'])} pool questions from {clusters} clusters "
          f"(of {len(with_evidence['pool'])}, at most {CLUSTER_CAP} per cluster). "
          f"Reporting on {len(questions['dev'])} development questions.")

    # --- The only paid step: embed the questions that have no cached vector ---------------
    texts = [q.text for split in questions.values() for q in split]
    to_embed = pipeline.embedder.uncached(texts, kind="query")
    counter = tiktoken.get_encoding("cl100k_base")
    tokens = sum(len(counter.encode(text, disallowed_special=())) for text in to_embed)
    estimate = embedding_cost(pipeline.embedder.model, tokens)
    print(f"To embed: {len(to_embed)} of {len(texts)} questions, about {tokens:,} tokens, "
          f"about ${estimate:.4f} with {pipeline.embedder.model}.")
    if args.estimate:
        raise SystemExit("Estimate only: nothing was embedded.")

    sweep = Sweep(pipeline, eval_cfg, questions)
    billed = 0
    for split in questions.values():  # also fills the cache that search_dense reads
        billed += pipeline.embedder.embed([q.text for q in split], kind="query").tokens
    if billed:
        cost = embedding_cost(pipeline.embedder.model, billed)
        append_ledger(LEDGER, "tune:stage05", pipeline.embedder.model, billed, 0, cost)
        print(f"Billed {billed:,} tokens, ${cost:.4f}.")
    sweep.search_dense(max(DEPTHS))
    for split in questions:
        dense = sweep.record(split, "dense", sweep.dense)
        print(f"dense, {split}: {pct(dense)}")

    # --- 1. BM25 alone: tokeniser, k1 and b ------------------------------------------------
    for tokeniser in TOKENISERS:
        retriever = SparseRetriever(**tokeniser)
        for k1, b in product(K1, B):
            retriever.k1, retriever.b = k1, b
            for split in questions:
                ranked = sweep.search_sparse(retriever, split, eval_cfg.depth)
                sweep.record(split, "sparse", ranked, **tokeniser, header=False, k1=k1, b=b)
    chosen = sweep.best("sparse")
    bm25 = {key: chosen[key] for key in ("stopwords", "stemmer", "k1", "b")}
    print(f"\nBM25 chosen on the pool: {bm25}\n  pool: {pct(chosen)}")
    print("\nBest k1 and b for each tokeniser, on the pool, and the same setting on dev:")
    for tokeniser in TOKENISERS:
        rows = [r for r in sweep.rows if r["family"] == "sparse"
                and all(r[key] == value for key, value in tokeniser.items())]
        top = max((r for r in rows if r["split"] == "pool"), key=lambda r: r["recall_2000tok"])
        on_dev = next(r for r in rows if r["split"] == "dev"
                      and (r["k1"], r["b"]) == (top["k1"], top["b"]))
        default = next(r for r in rows if r["split"] == "pool" and (r["k1"], r["b"]) == (1.2, 0.75))
        print(f"  {tokeniser}: k1 {top['k1']}, b {top['b']}: pool "
              f"{100 * top['recall_2000tok']:.1f} (at k1 1.2, b 0.75: "
              f"{100 * default['recall_2000tok']:.1f}), dev {100 * on_dev['recall_2000tok']:.1f}")

    # --- 2. Fusion, with the chosen BM25 settings -------------------------------------------
    retriever = SparseRetriever(**bm25)
    sparse = {
        split: sweep.search_sparse(retriever, split, max(DEPTHS)) for split in questions
    }

    def fused(split: str, depth: int, method: str, **settings) -> dict:
        return {
            q.question_id: fuse(
                sweep.dense[q.question_id], sparse[split][q.question_id], depth, method,
                **settings,
            )
            for q in questions[split]
        }

    for split in questions:
        for k in RRF_K:
            sweep.record(split, "rrf", fused(split, DEPTH, "rrf", rrf_k=k),
                         method="rrf", rrf_k=k, depth=DEPTH)
        for normalise, weight in product(NORMALISERS, WEIGHTS):
            sweep.record(
                split, "weighted",
                fused(split, DEPTH, "weighted", weight=weight, normalise=normalise),
                method="weighted", weight=weight, normalise=normalise, depth=DEPTH,
            )
    rrf, weighted = sweep.best("rrf"), sweep.best("weighted")
    print(f"\nRRF constant chosen on the pool: {rrf['rrf_k']}\n  pool: {pct(rrf)}")
    print(f"Weighted fusion chosen on the pool: dense weight {weighted['weight']}, "
          f"{weighted['normalise']}\n  pool: {pct(weighted)}")
    for family, key in (("rrf", "rrf_k"), ("weighted", "weight")):
        print(f"\n{family} sweep, recall in 2,000 tok (pool / dev):")
        for normalise in NORMALISERS if family == "weighted" else [None]:
            rows = [r for r in sweep.rows if r["family"] == family
                    and r.get("normalise") == normalise]
            for value in sorted({r[key] for r in rows}):
                pool, dev = (
                    next(r for r in rows if r["split"] == s and r[key] == value)
                    for s in ("pool", "dev")
                )
                label = f"{normalise} " if normalise else ""
                print(f"  {label}{key} {value}: {100 * pool['recall_2000tok']:.1f} / "
                      f"{100 * dev['recall_2000tok']:.1f}")

    # --- 3. How deep each retriever searches before fusion ----------------------------------
    print("\nDepth before fusion, recall in 2,000 tok on the pool (RRF / weighted):")
    for depth in DEPTHS:
        results = []
        for split in questions:
            a = sweep.record(split, "depth", fused(split, depth, "rrf", rrf_k=rrf["rrf_k"]),
                             method="rrf", rrf_k=rrf["rrf_k"], depth=depth)
            b = sweep.record(
                split, "depth",
                fused(split, depth, "weighted", weight=weighted["weight"],
                      normalise=weighted["normalise"]),
                method="weighted", weight=weighted["weight"],
                normalise=weighted["normalise"], depth=depth,
            )
            if split == "pool":
                results = [a, b]
        print(f"  {depth}: {100 * results[0]['recall_2000tok']:.1f} / "
              f"{100 * results[1]['recall_2000tok']:.1f}")

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
