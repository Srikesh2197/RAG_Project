"""The pipeline: build components from a config, ingest documents, answer questions.

    Index time:  documents -> chunker -> embedder -> store (saved to disk)
    Query time:  question -> embedder -> store.search -> prompt -> generator -> Trace
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

# Importing a component module registers it by name.
import ragbasics.chunking.fixed
import ragbasics.chunking.parent_child
import ragbasics.chunking.recursive
import ragbasics.chunking.semantic
import ragbasics.embedding.hashing
import ragbasics.embedding.local
import ragbasics.embedding.openai_embedder
import ragbasics.generation.llm
import ragbasics.index.numpy_store  # noqa: F401
from ragbasics.config import ComponentSpec, PipelineConfig
from ragbasics.context.prompts import (
    CLOSED_BOOK_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_closed_book_prompt,
    build_user_prompt,
)
from ragbasics.costs import append_ledger, embedding_cost, generation_cost
from ragbasics.embedding.cache import EmbeddingCache
from ragbasics.registry import build
from ragbasics.types import Candidate, Chunk, Document, Trace

EMBED_BATCH = 100  # chunks per embedding request


class IndexMismatchError(Exception):
    """The index on disk was built with a different chunker or embedder than the config."""


@dataclass(frozen=True)
class IngestReport:
    documents: int
    skipped: int  # already in the index
    chunks: int
    embed_tokens: int
    cost_usd: float
    seconds: float
    chunk_seconds: float = 0.0  # the part of `seconds` spent chunking
    chunker_tokens: int = 0  # embedding tokens the chunker itself used (semantic)


def _build(kind: str, spec: ComponentSpec) -> Any:
    return build(kind, spec.name, **spec.params)


def collapse_duplicates(candidates: list[Candidate]) -> list[Candidate]:
    """Keep the best-ranked candidate for each returned span, and renumber the ranks.

    Flat chunkers never return one span twice, so this changes nothing for them. With
    parent-child chunking several children share a parent, and the parent's text should
    reach the prompt once.
    """
    seen: set[tuple[str, int, int]] = set()
    kept: list[Candidate] = []
    for candidate in candidates:
        span = (candidate.chunk.doc_id, candidate.chunk.start_char, candidate.chunk.end_char)
        if span not in seen:
            seen.add(span)
            kept.append(replace(candidate, rank=len(kept) + 1))
    return kept


class Pipeline:
    def __init__(
        self, cfg: PipelineConfig, index_dir: Path | None = None, ledger: Path | None = None
    ):
        self.cfg = cfg
        self.index_dir = index_dir or cfg.index_dir
        self.ledger = ledger
        self.chunker = _build("chunker", cfg.chunker)
        self.embedder = _build("embedder", cfg.embedder)
        if cfg.embedding_cache and hasattr(self.embedder, "cache"):
            self.embedder.cache = EmbeddingCache(cfg.embedding_cache)
        self.store = _build("store", cfg.store)
        self.generator = _build("generator", cfg.generator)

    # --- Index time ------------------------------------------------------------------

    def fingerprint(self) -> dict[str, Any]:
        """What the index depends on. Vectors from one embedder mean nothing to another,
        and chunks from one chunker are not the chunks of another."""
        return {
            "chunker": self.cfg.chunker.model_dump(),
            "embedder": self.cfg.embedder.model_dump(),
        }

    def chunk(self, documents: list[Document]) -> list[Chunk]:
        # A chunker that calls a model (semantic) does it for all documents in one go.
        if hasattr(self.chunker, "prepare"):
            self.chunker.prepare(documents)
        return [chunk for document in documents for chunk in self.chunker.chunk(document)]

    def ingest(
        self,
        documents: list[Document],
        progress: Callable[[int, int], None] | None = None,
    ) -> IngestReport:
        """Chunk, embed and store `documents`, then save the index.

        Documents whose id is already in the index are skipped, so running it twice
        does not create duplicates. `progress(done, total)` is called after each batch.
        """
        started = time.perf_counter()
        indexed = {chunk.doc_id for chunk in self.store.chunks}
        new = [d for d in documents if d.doc_id not in indexed]
        chunker_tokens_before = getattr(self.chunker, "billed_tokens", 0)
        chunks = self.chunk(new)
        chunk_seconds = time.perf_counter() - started
        chunker_tokens = getattr(self.chunker, "billed_tokens", 0) - chunker_tokens_before

        tokens = 0
        for i in range(0, len(chunks), EMBED_BATCH):
            batch = chunks[i : i + EMBED_BATCH]
            embedded = self.embedder.embed([chunk.text_to_embed for chunk in batch])
            self.store.add(batch, embedded.vectors)
            tokens += embedded.tokens
            if progress:
                progress(min(i + EMBED_BATCH, len(chunks)), len(chunks))

        self.store.save(self.index_dir, self.fingerprint())
        cost = embedding_cost(self.embedder.model, tokens)
        if self.ledger and tokens:
            append_ledger(self.ledger, "ingest", self.embedder.model, tokens, 0, cost)
        if chunker_tokens:
            chunker_cost = embedding_cost(self.chunker.model, chunker_tokens)
            cost += chunker_cost
            if self.ledger:
                append_ledger(
                    self.ledger, "chunk", self.chunker.model, chunker_tokens, 0, chunker_cost
                )
        return IngestReport(
            documents=len(new),
            skipped=len(documents) - len(new),
            chunks=len(chunks),
            embed_tokens=tokens,
            cost_usd=cost,
            seconds=time.perf_counter() - started,
            chunk_seconds=chunk_seconds,
            chunker_tokens=chunker_tokens,
        )

    def load(self) -> bool:
        """Load the index from disk. Returns False if there is none yet."""
        if not (self.index_dir / "meta.json").exists():
            return False
        store, meta = type(self.store).load(self.index_dir)
        if meta != self.fingerprint():
            raise IndexMismatchError(
                f"The index in {self.index_dir} was built with {meta}, but the config asks "
                f"for {self.fingerprint()}. Delete the index directory and ingest again."
            )
        self.store = store
        return True

    # --- Query time ------------------------------------------------------------------

    def retrieve(self, question: str, k: int | None = None) -> Trace:
        """Embed the question and search. `k` defaults to the config's `top_k`; the
        evaluator asks for more, to score ranks below the ones that reach the prompt."""
        if len(self.store) == 0:
            raise ValueError("The index is empty. Ingest documents first.")
        trace = Trace(
            question=question,
            models={"embedder": self.embedder.model, "generator": self.generator.model},
        )
        t0 = time.perf_counter()
        embedded = self.embedder.embed([question], kind="query")
        t1 = time.perf_counter()
        k = k or self.cfg.top_k
        found = self.store.search(embedded.vectors[0], k * self.cfg.overfetch)
        trace.candidates = collapse_duplicates(found)[:k]
        t2 = time.perf_counter()
        trace.timings = {"embed_query": t1 - t0, "search": t2 - t1}
        trace.usage = {"embed_tokens": embedded.tokens}
        trace.cost_usd = embedding_cost(self.embedder.model, embedded.tokens)
        return trace

    def generate(self, trace: Trace, context: list[Chunk] | None) -> Trace:
        """Build the prompt from `context` and generate the answer, filling in `trace`.

        `context=None` asks the question closed-book, with no passages at all.
        """
        if context is None:
            trace.context = []
            trace.system_prompt = CLOSED_BOOK_SYSTEM_PROMPT
            trace.user_prompt = build_closed_book_prompt(trace.question)
        else:
            trace.context = context
            trace.system_prompt = SYSTEM_PROMPT
            trace.user_prompt = build_user_prompt(trace.question, context)

        t0 = time.perf_counter()
        generation = self.generator.generate(trace.system_prompt, trace.user_prompt)
        trace.timings["generate"] = time.perf_counter() - t0

        trace.answer = generation.text
        trace.stop_reason = generation.stop_reason
        trace.cached = generation.cached
        trace.usage |= {
            "input_tokens": generation.input_tokens,
            "output_tokens": generation.output_tokens,
        }
        trace.cost_usd += generation_cost(
            self.generator.model, generation.input_tokens, generation.output_tokens
        )
        return trace

    def ask(self, question: str) -> Trace:
        trace = self.retrieve(question)
        trace = self.generate(trace, [c.chunk for c in trace.candidates])
        if self.ledger and trace.cost_usd:
            append_ledger(
                self.ledger,
                "ask",
                self.generator.model,
                trace.usage["input_tokens"],
                trace.usage["output_tokens"],
                trace.cost_usd,
            )
        return trace
