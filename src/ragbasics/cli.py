"""Command line: rag ingest | ask | check."""

import argparse
import os
import sys
from pathlib import Path

import tiktoken
from dotenv import load_dotenv

from ragbasics.config import PipelineConfig, load_config
from ragbasics.costs import embedding_cost
from ragbasics.eval.dataset import read_documents
from ragbasics.pipeline import Pipeline
from ragbasics.types import Trace

DEFAULT_CONFIG = "configs/baseline.yaml"
DEFAULT_DOCUMENTS = "data/processed/documents.jsonl"
LEDGER = Path("runs/cost_ledger.csv")


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
    chunks = pipeline.chunk([d for d in documents if d.doc_id not in indexed])
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = sum(len(encoding.encode(c.text_to_embed, disallowed_special=())) for c in chunks)
    estimate = embedding_cost(pipeline.embedder.model, tokens)
    print(
        f"{len(documents)} documents, {len(indexed)} already indexed. "
        f"To embed: {len(chunks)} chunks, about {tokens:,} tokens, "
        f"about ${estimate:.4f} with {pipeline.embedder.model}."
    )

    def progress(done: int, total: int) -> None:
        print(f"\r  embedded {done}/{total} chunks", end="", flush=True)

    report = pipeline.ingest(documents, progress=progress)
    print(
        f"\nIndexed {report.documents} documents as {report.chunks} chunks in "
        f"{report.seconds:.1f} s. Billed {report.embed_tokens:,} tokens, ${report.cost_usd:.4f}. "
        f"Index: {pipeline.index_dir} ({len(pipeline.store)} chunks)."
    )


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
        # The judge (Stage 2) will be picked from these.
        print("OpenAI gpt models:", ", ".join(i for i in ids if i.startswith("gpt")))

    if os.environ.get("ANTHROPIC_API_KEY"):
        import anthropic

        generator = cfg.generator.params.get("model", "")
        try:
            model = anthropic.Anthropic().models.retrieve(generator)
            print(f"Anthropic generator '{generator}': found ({model.display_name})")
        except anthropic.NotFoundError:
            print(f"Anthropic generator '{generator}': NOT FOUND")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(prog="rag")
    commands = parser.add_subparsers(dest="command", required=True)

    ingest = commands.add_parser("ingest", help="chunk, embed and index documents")
    ingest.add_argument("--config", default=DEFAULT_CONFIG)
    ingest.add_argument("--documents", default=DEFAULT_DOCUMENTS)
    ingest.set_defaults(run=cmd_ingest)

    ask = commands.add_parser("ask", help="answer one question from the index")
    ask.add_argument("question")
    ask.add_argument("--config", default=DEFAULT_CONFIG)
    ask.add_argument("--show-prompt", action="store_true")
    ask.set_defaults(run=cmd_ask)

    check = commands.add_parser("check", help="check API keys and model ids")
    check.add_argument("--config", default=DEFAULT_CONFIG)
    check.set_defaults(run=cmd_check)

    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
