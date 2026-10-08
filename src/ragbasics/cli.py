"""Command line: rag ingest | ask | check | eval | report | compare."""

import argparse
import json
import os
import sys
from pathlib import Path

import tiktoken
from dotenv import load_dotenv

from ragbasics.config import EvalConfig, PipelineConfig, load_config
from ragbasics.costs import embedding_cost
from ragbasics.eval import report
from ragbasics.eval.dataset import read_documents
from ragbasics.eval.runner import MODES, ConfirmationNeeded, EstimateOnly, run_eval
from ragbasics.pipeline import Pipeline
from ragbasics.types import Trace

DEFAULT_CONFIG = "configs/baseline.yaml"
DEFAULT_DOCUMENTS = "data/processed/documents.jsonl"
DEFAULT_EVAL_CONFIG = "configs/eval.yaml"
RUNS = Path("runs")
LEDGER = RUNS / "cost_ledger.csv"
RESULTS_CSV = Path("results/results.csv")
README = Path("README.md")


def source_label(metadata: dict) -> str:
    """'TechCrunch, 2023-10-07: Title' from whichever of those the chunk carries."""
    head = ", ".join(
        str(part)[:10] if key == "published_at" else str(part)
        for key in ("source", "published_at")
        if (part := metadata.get(key))
    )
    title = metadata.get("title") or ""
    return ": ".join(part for part in (head, title) if part)


def format_trace(trace: Trace) -> str:
    lines = [f"Answer: {trace.answer}", "", "Retrieved chunks:"]
    for c in trace.candidates:
        chunk = c.chunk
        lines.append(
            f"  {c.rank}. score {c.score:.3f}  {chunk.chunk_id} "
            f"[{chunk.start_char}:{chunk.end_char}]  {source_label(chunk.metadata)}"
        )
    timings = ", ".join(f"{step} {s * 1000:.0f} ms" for step, s in trace.timings.items())
    lines += ["", f"Time: {timings}", f"Tokens: {trace.usage}  Cost: ${trace.cost_usd:.4f}"]
    return "\n".join(lines)


def cmd_ingest(args: argparse.Namespace) -> None:
    pipeline = Pipeline(load_config(args.config, PipelineConfig), ledger=LEDGER)
    pipeline.load()
    documents = read_documents(Path(args.documents))

    indexed = {chunk.doc_id for chunk in pipeline.store.chunks}
    new = [d for d in documents if d.doc_id not in indexed]
    encoding = tiktoken.get_encoding("cl100k_base")

    def count(texts: list[str]) -> int:
        return sum(len(encoding.encode(text, disallowed_special=())) for text in texts)

    # A chunker that embeds sentences (semantic) spends before any chunk exists, so its
    # cost is printed first.
    chunker = pipeline.chunker
    if hasattr(chunker, "texts_to_embed"):
        tokens = count(chunker.texts_to_embed(new))
        print(
            f"The {pipeline.cfg.chunker.name} chunker embeds sentences first: about "
            f"{tokens:,} tokens, about ${embedding_cost(chunker.model, tokens):.4f} with "
            f"{chunker.model} (0 if cached)."
        )
        if args.estimate:
            sys.exit("Estimate only: the chunk embedding costs about as much as one index "
                     "build of the same text. Nothing was embedded.")

    chunks = pipeline.chunk(new)
    texts = [c.text_to_embed for c in chunks]
    # Vectors in the embedding cache are not computed or paid for again.
    embedder = pipeline.embedder
    to_embed = embedder.uncached(texts) if hasattr(embedder, "uncached") else texts
    tokens = count(to_embed)
    estimate = embedding_cost(embedder.model, tokens)
    print(
        f"{len(documents)} documents, {len(indexed)} already indexed. "
        f"{len(chunks)} chunks holding {len(set(texts))} distinct texts, "
        f"{len(set(texts)) - len(set(to_embed))} of them with a cached vector. "
        f"To embed: {len(to_embed)} texts, about {tokens:,} tokens, "
        f"about ${estimate:.4f} with {embedder.model}."
    )
    if args.estimate:
        sys.exit("Estimate only: nothing was embedded.")

    def progress(done: int, total: int) -> None:
        print(f"\r  embedded {done}/{total} chunks", end="", flush=True)

    report = pipeline.ingest(documents, progress=progress)
    print(
        f"\nIndexed {report.documents} documents as {report.chunks} chunks in "
        f"{report.seconds:.1f} s. Billed {report.embed_tokens:,} tokens, ${report.cost_usd:.4f}. "
        f"Index: {pipeline.index_dir} ({len(pipeline.store)} chunks)."
    )
    if report.documents:
        # Kept next to the index so a stage write-up can report indexing time and cost.
        stats = {
            "documents": report.documents,
            "chunks": report.chunks,
            "embed_tokens": report.embed_tokens,
            "chunker_tokens": report.chunker_tokens,
            "cost_usd": round(report.cost_usd, 6),
            "seconds": round(report.seconds, 1),
            "chunk_seconds": round(report.chunk_seconds, 1),
        }
        (pipeline.index_dir / "ingest.json").write_text(json.dumps(stats, indent=2) + "\n")


def cmd_ask(args: argparse.Namespace) -> None:
    pipeline = Pipeline(load_config(args.config, PipelineConfig), ledger=LEDGER)
    if not pipeline.load():
        sys.exit(f"No index in {pipeline.index_dir}. Run `rag ingest` first.")
    trace = pipeline.ask(args.question)
    print(format_trace(trace))
    if args.show_prompt:
        print(f"\n--- system ---\n{trace.system_prompt}\n\n--- user ---\n{trace.user_prompt}")


def cmd_check(args: argparse.Namespace) -> None:
    """Confirm both API keys work and the configured model ids exist."""
    cfg = load_config(args.config, PipelineConfig)
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        print(f"{key}: {'set' if os.environ.get(key) else 'MISSING'}")

    if os.environ.get("OPENAI_API_KEY"):
        from openai import OpenAI

        client = OpenAI()
        ids = sorted(model.id for model in client.models.list())
        embedder = cfg.embedder.params.get("model", "")
        print(f"OpenAI embedder '{embedder}': {'found' if embedder in ids else 'NOT FOUND'}")
        # The judge (configs/eval.yaml) is picked from these.
        print("OpenAI gpt models:", ", ".join(i for i in ids if i.startswith("gpt")))

    if os.environ.get("ANTHROPIC_API_KEY"):
        import anthropic

        generator = cfg.generator.params.get("model", "")
        try:
            model = anthropic.Anthropic().models.retrieve(generator)
            print(f"Anthropic generator '{generator}': found ({model.display_name})")
        except anthropic.NotFoundError:
            print(f"Anthropic generator '{generator}': NOT FOUND")


def cmd_eval(args: argparse.Namespace) -> None:
    try:
        run_dir = run_eval(
            load_config(args.config, PipelineConfig),
            load_config(args.eval_config, EvalConfig),
            mode=args.mode,
            sample=args.sample,
            answers=not args.retrieval_only,
            confirmed=args.yes,
            estimate_only=args.estimate,
            runs_dir=RUNS,
            ledger=LEDGER,
        )
    except (ConfirmationNeeded, EstimateOnly) as stop:
        sys.exit(str(stop))
    print(report.results_markdown([report.load_run(run_dir)]))


def cmd_report(args: argparse.Namespace) -> None:
    runs = report.load_runs(RUNS)
    if not runs:
        sys.exit(f"No runs in {RUNS}/. Run `rag eval` first.")
    table = report.results_markdown(runs)
    report.write_csv(runs, RESULTS_CSV)
    in_readme = report.update_readme(README, table)
    print(table)
    print(f"\nWrote {RESULTS_CSV}" + (f" and the table in {README}." if in_readme else "."))


def cmd_compare(args: argparse.Namespace) -> None:
    """Paired difference between two runs, metric by metric."""
    a, b = report.load_run(RUNS / args.run_a), report.load_run(RUNS / args.run_b)
    print(f"{args.run_a} minus {args.run_b}, percentage points, 95% interval over clusters")
    for metric in a["metrics"]:
        if metric not in b["metrics"]:
            continue
        diff = report.compare(a, b, metric)
        changed, shared = report.flips(a, b, metric)
        print(
            f"  {metric:32s} {100 * diff.mean:+6.1f}  ({100 * diff.low:+.1f} to "
            f"{100 * diff.high:+.1f})   {changed} of {shared} questions differ"
        )


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="rag")
    commands = parser.add_subparsers(dest="command", required=True)

    ingest = commands.add_parser("ingest", help="chunk, embed and index documents")
    ingest.add_argument("--config", default=DEFAULT_CONFIG)
    ingest.add_argument("--documents", default=DEFAULT_DOCUMENTS)
    ingest.add_argument("--estimate", action="store_true", help="print the cost and stop")
    ingest.set_defaults(run=cmd_ingest)

    ask = commands.add_parser("ask", help="answer one question from the index")
    ask.add_argument("question")
    ask.add_argument("--config", default=DEFAULT_CONFIG)
    ask.add_argument("--show-prompt", action="store_true")
    ask.set_defaults(run=cmd_ask)

    check = commands.add_parser("check", help="check API keys and model ids")
    check.add_argument("--config", default=DEFAULT_CONFIG)
    check.set_defaults(run=cmd_check)

    evaluate = commands.add_parser("eval", help="score a pipeline on the development questions")
    evaluate.add_argument("--config", default=DEFAULT_CONFIG)
    evaluate.add_argument("--eval-config", default=DEFAULT_EVAL_CONFIG)
    evaluate.add_argument("--mode", choices=MODES, default="retrieved")
    evaluate.add_argument("--sample", type=int, default=0, help="1 repeats the generator calls")
    evaluate.add_argument("--retrieval-only", action="store_true", help="no LLM calls")
    evaluate.add_argument("--estimate", action="store_true", help="print the cost and stop")
    evaluate.add_argument("--yes", action="store_true", help="go ahead above the cost limit")
    evaluate.set_defaults(run=cmd_eval)

    rep = commands.add_parser("report", help="rebuild results/results.csv and the README table")
    rep.set_defaults(run=cmd_report)

    comp = commands.add_parser("compare", help="paired difference between two runs")
    comp.add_argument("run_a")
    comp.add_argument("run_b")
    comp.set_defaults(run=cmd_compare)

    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
