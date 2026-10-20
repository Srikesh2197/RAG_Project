"""Stage 7: write the query transformations for every question, once, into the LLM cache.

    python scripts/stage07_transform.py --examples 5    # five outputs each; about a cent
    python scripts/stage07_transform.py --estimate      # the cost of the full set, and stop
    python scripts/stage07_transform.py                 # write them; every call is cached
    python scripts/stage07_transform.py --effort low    # the same with less reasoning

Four transformations call an LLM once per question: rewrite, multi-query, HyDE and
decomposition. Their settings (model, prompt, limits) are read from
configs/stage07_query/, so what is written here is exactly what `rag eval` and
scripts/stage07_tune.py will ask for, and those then cost nothing.

Questions: all 500 development questions (`rag eval` transforms every one, including the
59 with no gold evidence) and the 284 pool questions the settings are chosen on.

The model bills its reasoning as output tokens, on top of the reply. `--examples` measures
output tokens on real calls and prints them; the means are the `output_tokens` figures in
src/ragbasics/query/llm.py, which every estimate uses.

A refused prompt falls back to the question as asked and is counted. Spend is written to
the ledger when the script ends, however it ends.
"""

import argparse
import statistics
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import tiktoken
from dotenv import load_dotenv

from ragbasics.config import PipelineConfig, load_config
from ragbasics.costs import generation_cost
from ragbasics.eval.dataset import read_questions
from ragbasics.query.base import Transformed
from ragbasics.query.clauses import split_clauses
from ragbasics.registry import build
from ragbasics.types import Question
from stage05_tune import CLUSTER_CAP, SEED, capped

CONFIGS = Path("configs/stage07_query")
LLM_TRANSFORMS = ["rewrite", "multi-query", "hyde", "decompose"]
SPLITS = {
    "dev": Path("data/processed/questions_dev.jsonl"),
    "pool": Path("data/processed/questions_pool.jsonl"),
}
LEDGER = Path("runs/cost_ledger.csv")
WORKERS = 6


def load_questions(splits: list[str]) -> dict[str, list[Question]]:
    out = {}
    for split in splits:
        questions = read_questions(SPLITS[split])
        if split == "pool":
            questions = capped([q for q in questions if q.evidence], CLUSTER_CAP, SEED)
        out[split] = questions
    return out


def transformer_for(key: str, effort: str | None):
    """The transformer of configs/stage07_query/<key>.yaml, optionally with another
    reasoning effort."""
    spec = load_config(CONFIGS / f"{key}.yaml", PipelineConfig).query_transform
    params = dict(spec.params)
    if effort:
        generator = dict(params["generator"])
        generator["params"] = dict(generator["params"]) | {"reasoning_effort": effort}
        params["generator"] = generator
    return build("query_transform", spec.name, **params)


def estimate(transformer, questions: list[Question], encoding) -> tuple[int, int, int, float]:
    """(new calls, tokens in, tokens out, dollars) for the questions not yet cached."""
    new = [q for q in questions if not transformer.is_cached(q.text)]
    tokens_in = sum(
        len(encoding.encode(transformer.system_prompt() + transformer.prompt(q.text)))
        for q in new
    )
    tokens_out = len(new) * transformer.output_tokens
    return len(new), tokens_in, tokens_out, generation_cost(transformer.model, tokens_in,
                                                            tokens_out)


def show(question: Question, result: Transformed) -> None:
    flags = {k: v for k, v in result.notes.items() if k != "stop_reason"}
    print(f"    {result.input_tokens} tokens in, {result.output_tokens} out"
          + (f", {flags}" if flags else ""))
    for query in result.queries:
        print(f"    [{query.role}] {query.text}")


def examples(n: int, effort: str | None, encoding) -> None:
    """The first question of each type, then more in file order, up to `n`."""
    dev = read_questions(SPLITS["dev"])
    picked: list[Question] = []
    for kind in ("comparison_query", "inference_query", "temporal_query", "null_query"):
        picked.append(next(q for q in dev if q.question_type == kind))
    picked += [q for q in dev if q not in picked]
    picked = picked[:n]
    transformers = {key: transformer_for(key, effort) for key in LLM_TRANSFORMS}
    total = sum(estimate(t, picked, encoding)[3] for t in transformers.values())
    print(f"{n} questions x {len(transformers)} transformations, reasoning effort "
          f"{effort or 'default'}. Estimated cost: ${total:.3f}.\n")
    spent: dict[str, list[int]] = {key: [] for key in transformers}
    try:
        for question in picked:
            print(f"{question.question_id} ({question.question_type}): {question.text}")
            print(f"  clauses (rule, no model): {split_clauses(question.text)}")
            for key, transformer in transformers.items():
                result = transformer.transform(question.text)
                print(f"  {key}:")
                show(question, result)
                spent[key].append(result.output_tokens)
            print()
    finally:
        billed = sum(t.meter.flush(LEDGER, "stage07:examples") for t in transformers.values())
    print("Output tokens per call, reply and reasoning together: median, mean, and the "
          "figure the estimate uses.")
    for key, counts in spent.items():
        if counts:
            print(f"  {key:12s} {statistics.median(counts):6.0f} {statistics.fmean(counts):6.0f} "
                  f"{transformers[key].output_tokens:6d}")
    print(f"Billed now: ${billed:.4f} (cached calls are free).")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", type=int, default=0, help="show this many and stop")
    parser.add_argument("--estimate", action="store_true", help="print the cost and stop")
    parser.add_argument("--effort", default=None, help="reasoning effort to send, e.g. low")
    parser.add_argument("--splits", nargs="+", choices=list(SPLITS), default=list(SPLITS))
    parser.add_argument("--only", nargs="+", choices=LLM_TRANSFORMS, default=LLM_TRANSFORMS)
    args = parser.parse_args()
    load_dotenv()
    encoding = tiktoken.get_encoding("cl100k_base")
    if args.examples:
        examples(args.examples, args.effort, encoding)
        return

    questions = load_questions(args.splits)
    everyone = [q for split in questions.values() for q in split]
    transformers = {key: transformer_for(key, args.effort) for key in args.only}
    print(", ".join(f"{len(qs)} {split} questions" for split, qs in questions.items()) + ".")
    total = 0.0
    for key, transformer in transformers.items():
        calls, tokens_in, tokens_out, dollars = estimate(transformer, everyone, encoding)
        total += dollars
        print(f"  {key:12s} {transformer.model}: {calls} new calls of {len(everyone)}, about "
              f"{tokens_in:,} tokens in and {tokens_out:,} out, ${dollars:.2f}")
    print(f"Estimated cost: ${total:.2f}.")
    if args.estimate:
        raise SystemExit("Estimate only: no question was transformed.")

    try:
        for key, transformer in transformers.items():
            with ThreadPoolExecutor(max_workers=WORKERS) as pool:
                results = list(pool.map(transformer.transform, [q.text for q in everyone]))
            counts = [len(r.queries) for r in results]
            flags = {
                flag: sum(bool(r.notes.get(flag)) for r in results)
                for flag in ("refused", "empty", "fallback", "repeated", "extra", "missing")
            }
            cut = sum(r.notes.get("stop_reason") == "length" for r in results)
            fresh = [r.output_tokens for r in results if not r.cached and r.calls]
            print(f"\n{key}: queries per question median {statistics.median(counts)}, mean "
                  f"{statistics.fmean(counts):.2f}, max {max(counts)}. {flags}, cut off at the "
                  f"output limit {cut}.")
            if fresh:
                print(f"  {len(fresh)} paid calls, output tokens median "
                      f"{statistics.median(fresh):.0f}, mean {statistics.fmean(fresh):.0f}.")
    finally:
        billed = sum(t.meter.flush(LEDGER, "stage07:transform") for t in transformers.values())
        print(f"\nBilled now: ${billed:.4f}. Written to {LEDGER}.")


if __name__ == "__main__":
    main()
