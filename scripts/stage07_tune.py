"""Stage 7 tuning: how each query transformation is used, chosen on questions that are not
the ones the results are reported on.

    python scripts/stage07_tune.py --estimate                  # what is not cached yet, and stop
    python scripts/stage07_tune.py --estimate --embed          # embed first, for exact pair counts
    python scripts/stage07_tune.py                             # rows with no reranker
    python scripts/stage07_tune.py --rerank candidates         # also: in front of the reranker
    python scripts/stage07_tune.py --rerank split --depth 50   # also: reranker scores the parts
    python scripts/stage07_tune.py --rerank candidates --rerank-splits dev   # reranker on dev only
    python scripts/stage07_tune.py --splits pool --write       # one split; write the CSV

As in Stages 5 and 6, choices are made on `questions_pool.jsonl` (at most 25 questions per
evidence cluster, seed 0) and the development questions are scored with the same sweep
only to show how sensitive each choice is. The pool shares articles and gold sentences
with development, so this guards against fitting the wording of the reported questions
and not their evidence.

Three families of rows, each against its own reference:

    first       no reranker. Each transformation, searched with in place of the question
                or next to it, its lists combined by rank fusion or round robin.
                Reference: hybrid retrieval with the question as asked.
    candidates  the best `first` form of each transformation in front of Qwen3-Reranker,
                which still scores every candidate against the question as asked. The
                transformation can only change which candidates the reranker is given.
                Reference: the reranker behind plain hybrid retrieval, at the same depth.
    split       the candidates are plain hybrid retrieval's. The reranker scores each one
                against every part of the question (rule-based clauses, or an LLM's
                sub-questions), and the orderings are merged by round robin or by best
                score. Reference: as for `candidates`.

What this script pays for: embedding the transformed queries, a few cents at most. It
never calls an LLM: the transformations must already be in the cache
(scripts/stage07_transform.py). The reranker runs on this machine and costs time, not
money: a score is cached per (query, chunk), so every new query text is new pairs.
`--estimate` counts them and times a small batch to give hours.
"""

import argparse
import copy
import csv
import statistics
import time
from pathlib import Path

import tiktoken
from dotenv import load_dotenv

from ragbasics.config import EvalConfig, PipelineConfig, load_config
from ragbasics.costs import append_ledger, embedding_cost
from ragbasics.eval import retrieval_metrics as rm
from ragbasics.eval.dataset import read_questions
from ragbasics.eval.runner import count_relevant, score_retrieval
from ragbasics.eval.stats import paired_bootstrap
from ragbasics.pipeline import Pipeline
from ragbasics.query.base import route
from ragbasics.registry import build
from ragbasics.rerank.base import passage_text
from ragbasics.rerank.cache import pair_key
from ragbasics.types import Question, Trace
from stage05_tune import CLUSTER_CAP, SEED, MemoEncoding, capped

FIRST_STAGE = Path("configs/stage05_retrieval/hybrid-rrf.yaml")
RERANKER = Path("configs/stage06_rerank/qwen3-100.yaml")
CONFIGS = Path("configs/stage07_query")
EVAL_CONFIG = Path("configs/eval.yaml")
SPLITS = {
    "pool": Path("data/processed/questions_pool.jsonl"),
    "dev": Path("data/processed/questions_dev.jsonl"),
}
LEDGER = Path("runs/cost_ledger.csv")
CSV = Path("results/stage07_tuning.csv")

TRANSFORMS = ["rewrite", "multi-query", "hyde", "decompose", "clauses"]
SPLITTERS = ["clauses", "decompose"]  # the ones whose output is the question's parts
COMBINERS = [{"method": "rrf", "k": 60}, {"method": "round_robin"}]
TYPES = {"comparison_query": "comparison", "inference_query": "inference",
         "temporal_query": "temporal"}
DECIDES = "recall@2000tok"
METRICS = [DECIDES, "full_support@2000tok", "ndcg@10", "full_support@10", "recall@50"]
FIELDS = ["split", "family", "transform", "retrieve_with", "combine", "rerank_with",
          "rerank_combine", "depth", "questions", *METRICS, "candidate_recall",
          "vs_reference", "vs_reference_low", "vs_reference_high",
          *[f"{label}_{suffix}" for label in TYPES.values() for suffix in ("recall", "vs")],
          "queries_per_question", "fallbacks"]
PROGRESS_EVERY = 640  # pairs
BENCH_PAIRS = 96


class MemoRetriever:
    """The pipeline's retriever with each (query, depth) searched once. Variants of one
    transformation differ in how lists are combined, not in the lists."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.memo: dict[tuple[str, int, str], object] = {}

    def retrieve(self, question, k, index, kind="query"):
        key = (question, k, kind)
        if key not in self.memo:
            self.memo[key] = self.inner.retrieve(question, k, index, kind)
        return self.memo[key]


def transformer_for(key: str, **routing):
    spec = load_config(CONFIGS / f"{key}.yaml", PipelineConfig).query_transform
    params = {k: v for k, v in spec.params.items()
              if k not in ("retrieve_with", "rerank_with", "combine", "rerank_combine")}
    return build("query_transform", spec.name, **(params | routing))


class Sweep:
    def __init__(self, base: Pipeline, eval_cfg: EvalConfig, questions: dict[str, list[Question]]):
        self.base = base
        self.base.retriever = MemoRetriever(base.retriever)
        self.eval_cfg = eval_cfg
        self.questions = questions
        self.encoding = MemoEncoding()
        everyone = [q for split in questions.values() for q in split]
        self.total_relevant = count_relevant(base.store.chunks, everyone)
        self.rows: list[dict] = []
        self.scored: dict[tuple, dict[str, dict[str, float]]] = {}  # row key -> per question

    def pipeline(self, transformer=None, reranker=None) -> Pipeline:
        """The base pipeline with another transformation and reranker. The index, the
        embedder and the memoised retriever are shared."""
        variant = copy.copy(self.base)
        variant.transformer, variant.reranker = transformer, reranker
        return variant

    def first_stage(self, pipeline: Pipeline, split: str) -> dict[str, Trace]:
        depth = self.eval_cfg.depth
        return {q.question_id: pipeline.first_stage(q.text, depth) for q in self.questions[split]}

    def score(self, split: str, traces: dict[str, Trace]) -> dict[str, dict[str, float]]:
        out = {}
        for q in self.questions[split]:
            trace = traces[q.question_id]
            chunks = [c.chunk for c in trace.candidates[: self.eval_cfg.depth]]
            scores = score_retrieval(q, chunks, self.total_relevant[q.question_id],
                                     self.eval_cfg, self.encoding)
            scores["full_support@2000tok"] = float(scores[DECIDES] == 1.0)
            scores["recall@50"] = rm.evidence_recall(chunks, q.evidence, self.eval_cfg.depth)
            # How much of the evidence the reranker was handed, whatever it did with it.
            given = [c.chunk for c in (trace.retrieved or trace.candidates)]
            scores["candidate_recall"] = rm.evidence_recall(given, q.evidence, len(given))
            scores["queries"] = float(len(trace.queries) or 1)
            scores["fallback"] = float(bool(trace.transform.get("fallback")))
            out[q.question_id] = scores
        return out

    def record(self, split: str, traces: dict[str, Trace], reference: tuple | None,
               **labels) -> dict:
        questions = self.questions[split]
        scored = self.score(split, traces)
        key = (split, *labels.values())
        self.scored[key] = scored

        def mean(name: str, kind: str | None = None) -> float:
            return statistics.fmean(scored[q.question_id][name] for q in questions
                                    if kind is None or q.question_type == kind)

        def versus(kind: str | None = None):
            kept = [q for q in questions if kind is None or q.question_type == kind]
            return paired_bootstrap(
                [scored[q.question_id][DECIDES] for q in kept],
                [self.scored[reference][q.question_id][DECIDES] for q in kept],
                [q.cluster for q in kept],
            )

        row = {"split": split, **labels, "questions": len(questions),
               **{name: mean(name) for name in METRICS},
               "candidate_recall": mean("candidate_recall"),
               "queries_per_question": mean("queries"),
               "fallbacks": int(sum(scored[q.question_id]["fallback"] for q in questions))}
        if reference:
            d = versus()
            row |= {"vs_reference": d.mean, "vs_reference_low": d.low, "vs_reference_high": d.high}
        for kind, label in TYPES.items():
            if not any(q.question_type == kind for q in questions):
                row[f"{label}_recall"] = float("nan")  # only with --limit
                continue
            row[f"{label}_recall"] = mean(DECIDES, kind)
            if reference:
                row[f"{label}_vs"] = versus(kind).mean
        self.rows.append(row)
        return row

    # --- The reranker: count pairs, score them in bulk, then rerank from the cache -------

    def pairs(self, pipeline: Pipeline, traces: dict[str, Trace]) -> list[tuple[str, str]]:
        """Every (query, passage) pair `pipeline.rerank` would score for these traces."""
        reranker = pipeline.reranker
        out = []
        for trace in traces.values():
            head = trace.candidates[: reranker.depth]
            for query in trace.rerank_queries or [trace.question]:
                text = query if isinstance(query, str) else query.text
                out += [(text, passage_text(c.chunk, reranker.header)) for c in head]
        return out

    def uncached(self, reranker, pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
        keys = {pair_key(reranker.scorer.identity, q, p): (q, p) for q, p in pairs}
        known = reranker.cache.get_many(list(keys))
        return [pair for key, pair in keys.items() if key not in known]

    def rerank(self, pipeline: Pipeline, traces: dict[str, Trace], label: str) -> dict[str, Trace]:
        reranker = pipeline.reranker
        todo = self.uncached(reranker, self.pairs(pipeline, traces))
        started = time.perf_counter()
        for start in range(0, len(todo), PROGRESS_EVERY):
            reranker.scores(todo[start : start + PROGRESS_EVERY])
            done = min(start + PROGRESS_EVERY, len(todo))
            elapsed = time.perf_counter() - started
            print(f"\r  {label}: {done:,} of {len(todo):,} new pairs, {elapsed / 60:.1f} min, "
                  f"{done / elapsed:.1f} pairs/s", end="", flush=True)
        if todo:
            print()
        depth = self.eval_cfg.depth
        return {qid: pipeline.rerank(copy.copy(trace), depth) for qid, trace in traces.items()}


def line(row: dict) -> str:
    what = " ".join(str(row.get(k, "")) for k in ("transform", "retrieve_with", "combine",
                                                    "rerank_with", "rerank_combine"))
    versus = (f"{100 * row['vs_reference']:+5.1f} ({100 * row['vs_reference_low']:+.1f} to "
              f"{100 * row['vs_reference_high']:+.1f})" if "vs_reference" in row else "")
    types = "  ".join(f"{100 * row[f'{label}_recall']:5.1f}" for label in TYPES.values())
    return (f"{what:48s} " + "  ".join(f"{100 * row[m]:5.1f}" for m in METRICS)
            + f"  {100 * row['candidate_recall']:5.1f}  {types}  {versus}")


HEADER = (f"{'transformation, searched with, combined by':48s} rec2k   fs2k   nDCG  fs@10  "
          "rec50  given   comp  infer   temp  recall in 2,000 tok vs reference")


def embed_new_queries(sweep: Sweep, keys: list[str], estimate_only: bool) -> None:
    """Embed every transformed query that has no vector yet, in one go, and say what it
    costs first. A transformation whose LLM calls are not cached is reported and left out."""
    embedder = sweep.base.embedder
    encoding = tiktoken.get_encoding("cl100k_base")
    texts: list[str] = []
    for key in keys:
        transformer = transformer_for(key)
        for q in [q for split in sweep.questions.values() for q in split]:
            if hasattr(transformer, "is_cached") and not transformer.is_cached(q.text):
                continue
            texts += [query.text for query in transformer.transform(q.text).queries]
    new = sorted(set(embedder.uncached(texts, kind="query")))
    tokens = sum(len(encoding.encode(text, disallowed_special=())) for text in new)
    cost = embedding_cost(embedder.model, tokens)
    print(f"Queries to embed: {len(new):,} new of {len(set(texts)):,}, about {tokens:,} tokens, "
          f"${cost:.4f} with {embedder.model}.")
    if estimate_only or not new:
        return
    embedded = embedder.embed(new, kind="query")
    billed = embedding_cost(embedder.model, embedded.tokens)
    if billed:
        append_ledger(LEDGER, "stage07:tune", embedder.model, embedded.tokens, 0, billed)


def missing_llm_calls(sweep: Sweep, keys: list[str]) -> dict[str, int]:
    everyone = [q for split in sweep.questions.values() for q in split]
    out = {}
    for key in keys:
        transformer = transformer_for(key)
        if hasattr(transformer, "is_cached"):
            out[key] = sum(not transformer.is_cached(q.text) for q in everyone)
    return {key: n for key, n in out.items() if n}


def split_variants(keys: list[str]):
    for key in keys:
        for rerank_with in ("queries", "both"):
            for rerank_combine in ("round_robin", "max"):
                yield key, rerank_with, rerank_combine


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--estimate", action="store_true")
    parser.add_argument("--embed", action="store_true",
                        help="with --estimate: embed the new queries anyway (cents)")
    parser.add_argument("--rerank", nargs="+", choices=["candidates", "split"], default=[])
    parser.add_argument("--depth", type=int, default=100, help="candidates the reranker reads")
    parser.add_argument("--splits", nargs="+", choices=list(SPLITS), default=list(SPLITS))
    parser.add_argument("--rerank-splits", nargs="+", choices=list(SPLITS), default=list(SPLITS),
                        help="run the reranker rows on these only; the forms are still "
                             "chosen on the pool, with no reranker")
    parser.add_argument("--only", nargs="+", choices=TRANSFORMS, default=TRANSFORMS)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--csv", type=Path, default=CSV, help="where --write puts the rows")
    parser.add_argument("--limit", type=int, default=0,
                        help="only the first N questions of each split: a quick check of the "
                             "script, not a result")
    args = parser.parse_args()
    load_dotenv()

    eval_cfg = load_config(EVAL_CONFIG, EvalConfig)
    base = Pipeline(load_config(FIRST_STAGE, PipelineConfig))
    if not base.load():
        raise SystemExit(f"No index in {base.index_dir}.")
    with_evidence = {s: [q for q in read_questions(SPLITS[s]) if q.evidence] for s in args.splits}
    questions = {s: capped(qs, CLUSTER_CAP, SEED) if s == "pool" else qs
                 for s, qs in with_evidence.items()}
    if args.limit:
        questions = {s: qs[: args.limit] for s, qs in questions.items()}
    sweep = Sweep(base, eval_cfg, questions)
    print(", ".join(f"{len(qs)} {split} questions" for split, qs in questions.items())
          + ". Settings are chosen on the pool.")

    missing = missing_llm_calls(sweep, args.only)
    if missing:
        print(f"Not in the LLM cache yet (questions per transformation): {missing}. "
              "Run scripts/stage07_transform.py; this script calls no LLM.")
    ready = [key for key in args.only if key not in missing]
    embed_new_queries(sweep, ready, args.estimate and not args.embed)

    spec = load_config(RERANKER, PipelineConfig).reranker
    reranker = None
    if args.rerank or args.estimate:
        reranker = build("reranker", spec.name, **(spec.params | {"depth": args.depth}))

    if args.estimate:
        estimate(sweep, reranker, ready, [k for k in SPLITTERS if k in args.only], args.depth)
        return

    plain = sweep.pipeline()
    chosen: dict[str, tuple[str, dict | None]] = {}
    for split in questions:
        first = sweep.first_stage(plain, split)
        none = (split, "first", "none")
        sweep.record(split, first, None, family="first", transform="none")
        # --- first: no reranker ------------------------------------------------------------
        for key in ready:
            for retrieve_with in ("queries", "both"):
                for combine in COMBINERS:
                    transformer = transformer_for(key, retrieve_with=retrieve_with,
                                                  combine=combine)
                    traces = sweep.first_stage(sweep.pipeline(transformer), split)
                    several = any(len(t.queries) > 1 for t in traces.values())
                    if not several and combine != COMBINERS[0]:
                        continue  # one query each: nothing to combine, the same row again
                    sweep.record(split, traces, none, family="first", transform=key,
                                 retrieve_with=retrieve_with,
                                 combine=combine["method"] if several else "")
        if split == "pool" or "pool" not in questions:
            for key in ready:
                best = max((r for r in sweep.rows if r["split"] == split
                            and r["family"] == "first" and r["transform"] == key),
                           key=lambda r: r[DECIDES])
                combine = next((c for c in COMBINERS if c["method"] == best["combine"]), None)
                chosen[key] = (best["retrieve_with"], combine)

        if not args.rerank or split not in args.rerank_splits:
            continue
        reference = (split, "rerank", "none", args.depth)
        # The reference reads as deep as the reranker does, like every row it is compared
        # with. Reranking `first` would hand it only the plain list's top k.
        with_reranker = sweep.pipeline(reranker=reranker)
        reranked = sweep.rerank(with_reranker, sweep.first_stage(with_reranker, split),
                                f"{split} reference")
        sweep.record(split, reranked, None, family="rerank", transform="none", depth=args.depth)
        # --- candidates: the transformation changes what the reranker is given ------------
        if "candidates" in args.rerank:
            for key in ready:
                retrieve_with, combine = chosen[key]
                routing = {"retrieve_with": retrieve_with} | ({"combine": combine} if combine
                                                               else {})
                pipeline = sweep.pipeline(transformer_for(key, **routing), reranker)
                traces = {q.question_id: pipeline.first_stage(q.text, eval_cfg.depth)
                          for q in questions[split]}
                traces = sweep.rerank(pipeline, traces, f"{split} {key}")
                sweep.record(split, traces, reference, family="candidates", transform=key,
                             retrieve_with=retrieve_with,
                             combine=combine["method"] if combine else "", depth=args.depth)
        # --- split: the reranker scores against the question's parts -----------------------
        if "split" in args.rerank:
            for key, rerank_with, rerank_combine in split_variants(
                [k for k in SPLITTERS if k in ready]
            ):
                transformer = transformer_for(key, retrieve_with="original",
                                              rerank_with=rerank_with,
                                              rerank_combine=rerank_combine)
                pipeline = sweep.pipeline(transformer, reranker)
                traces = {q.question_id: pipeline.first_stage(q.text, eval_cfg.depth)
                          for q in questions[split]}
                traces = sweep.rerank(pipeline, traces, f"{split} {key} {rerank_with}")
                sweep.record(split, traces, reference, family="split", transform=key,
                             rerank_with=rerank_with, rerank_combine=rerank_combine,
                             depth=args.depth)

    for split in questions:
        for family in ("first", "rerank", "candidates", "split"):
            rows = [r for r in sweep.rows if r["split"] == split and r["family"] == family]
            if rows:
                print(f"\n{split} ({len(questions[split])} questions), {family}\n{HEADER}")
                print("\n".join(line(row) for row in rows))
    if chosen:
        print("\nForm of each transformation chosen on the pool by recall in 2,000 tokens, "
              "with no reranker:")
        for key, (retrieve_with, combine) in chosen.items():
            print(f"  {key}: searched with {retrieve_with}"
                  + (f", combined by {combine['method']}" if combine else ""))
    if args.write:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS, restval="")
            writer.writeheader()
            for row in sweep.rows:
                writer.writerow({k: round(v, 4) if isinstance(v, float) else v
                                 for k, v in row.items()})
        print(f"\nWrote {args.csv} ({len(sweep.rows)} rows)")


def estimate(sweep: Sweep, reranker, ready: list[str], splitters: list[str], depth: int) -> None:
    """New reranker pairs per family and split, and hours at a rate timed now on a small
    batch. Nothing is scored for the tables; the timed pairs go in the cache."""
    plain = sweep.pipeline(reranker=reranker)
    counts: list[tuple[str, str, int, bool]] = []  # (split, row, new pairs, exact)
    sample: list[tuple[str, str]] = []
    embedder = sweep.base.embedder
    for split, questions in sweep.questions.items():
        first = {q.question_id: plain.first_stage(q.text, sweep.eval_cfg.depth)
                 for q in questions}
        counts.append((split, "reference: the question as asked",
                       len(sweep.uncached(reranker, sweep.pairs(plain, first))), True))
        for key in splitters:
            if key not in ready:
                counts.append((split, f"split, {key}: not transformed yet, about",
                               int(len(questions) * depth * 2.6), False))
                continue
            transformer = transformer_for(key, retrieve_with="original", rerank_with="queries")
            pipeline = sweep.pipeline(transformer, reranker)
            traces = {}
            for q in questions:
                trace = copy.copy(first[q.question_id])
                parts = transformer.transform(q.text).queries
                trace.rerank_queries = route(q.text, parts, "queries")
                traces[q.question_id] = trace
            new = sweep.uncached(reranker, sweep.pairs(pipeline, traces))
            sample = sample or new
            counts.append((split, f"split, {key}", len(new), True))
        for key in TRANSFORMS:
            # Exact only once the queries have vectors: the candidates are not known before.
            transformer = transformer_for(key) if key in ready else None
            texts = ([query.text for q in questions
                      for query in transformer.transform(q.text).queries] if transformer else [])
            if transformer is None or embedder.uncached(texts, kind="query"):
                counts.append((split, f"candidates, {key}: at most", len(questions) * depth, False))
                continue
            traces = {q.question_id: sweep.pipeline(transformer, reranker).first_stage(
                q.text, sweep.eval_cfg.depth) for q in questions}
            new = sweep.uncached(reranker, sweep.pairs(plain, traces))
            counts.append((split, f"candidates, {key}", len(new), True))

    rate = 9.0
    if sample:
        batch = sample[:BENCH_PAIRS]
        reranker.scores(batch[:32])  # load the weights and warm the GPU
        started = time.perf_counter()
        reranker.scores(batch[32:])
        rate = len(batch[32:]) / (time.perf_counter() - started)
        print(f"Timed {len(batch) - 32} clause pairs: {rate:.1f} pairs per second "
              "(whole questions are longer: about 9 per second in Stage 6).")
    print(f"\nNew reranker pairs at depth {depth} (cached pairs cost nothing):")
    for split, row, pairs, exact in counts:
        speed = rate if row.startswith("split") else 9.0
        print(f"  {split:5s} {row:44s} {pairs:>8,} pairs  {pairs / speed / 3600:5.1f} h"
              + ("" if exact else "  (not exact)"))


if __name__ == "__main__":
    main()
