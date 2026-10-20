"""The pipeline: build components from a config, ingest documents, answer questions.

    Index time:  documents -> chunker -> embedder -> store (saved to disk)
    Query time:  question -> (query transformation) -> retriever -> (reranker)
                 -> prompt -> generator -> Trace

The retriever is dense (embed the question, search the vectors), sparse (BM25 over the
same chunks) or hybrid (both, fused). The optional reranker reads the question together
with each of the retriever's top candidates and reorders them. The optional query
transformation turns the question into one or more queries first; the retriever then
runs once per query and the lists are combined. The generator always answers the
question as asked.
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
import ragbasics.index.numpy_store
import ragbasics.query.clauses
import ragbasics.query.followup
import ragbasics.query.llm
import ragbasics.rerank.cross_encoder
import ragbasics.rerank.llm_listwise
import ragbasics.retrieval.dense
import ragbasics.retrieval.hybrid
import ragbasics.retrieval.sparse  # noqa: F401
from ragbasics.config import ComponentSpec, PipelineConfig
from ragbasics.context.prompts import (
    CLOSED_BOOK_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_closed_book_prompt,
    build_user_prompt,
)
from ragbasics.costs import append_ledger, embedding_cost, generation_cost
from ragbasics.embedding.cache import EmbeddingCache
from ragbasics.query.base import route
from ragbasics.query.followup import Turn
from ragbasics.registry import build
from ragbasics.rerank.base import merge
from ragbasics.retrieval.fusion import reciprocal_rank_fusion, round_robin
from ragbasics.types import Candidate, Chunk, Document, Query, Trace

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
        self.retriever = _build("retriever", cfg.retriever)
        self.reranker = _build("reranker", cfg.reranker) if cfg.reranker else None
        self.transformer = (
            _build("query_transform", cfg.query_transform) if cfg.query_transform else None
        )
        self.followup = _build("followup", cfg.followup) if cfg.followup else None
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
        """Rank chunks for the question: the configured retriever, then the reranker if
        there is one. `k` defaults to the config's `top_k`; the evaluator asks for more,
        to score ranks below the ones that reach the prompt."""
        return self.rerank(self.first_stage(question, k), k)

    def first_stage(self, question: str, k: int | None = None) -> Trace:
        """The retriever's list, before any reranking. With a reranker it is as deep as
        the reranker reads, when that is more than `k`."""
        if len(self.store) == 0:
            raise ValueError("The index is empty. Ingest documents first.")
        trace = Trace(
            question=question,
            models={"embedder": self.embedder.model, "generator": self.generator.model},
        )
        k = k or self.cfg.top_k
        fetch = max(k, self.reranker.depth) if self.reranker else k
        searches = self.transform(trace) if self.transformer else [Query(question)]
        # Several lists are fetched deeper than one would be, before they are combined.
        each = max(fetch, self.transformer.depth) if len(searches) > 1 else fetch
        # One search per query. A made-up answer passage is embedded as a passage.
        found = [
            self.retriever.retrieve(
                query.text,
                each * self.cfg.overfetch,
                self,
                kind="document" if query.role == "hypothetical" else "query",
            )
            for query in searches
        ]
        for result in found:
            for step, seconds in result.timings.items():
                trace.timings[step] = trace.timings.get(step, 0.0) + seconds
        if len(found) == 1:
            candidates = found[0].candidates
            trace.retrievers = found[0].lists
        else:
            t0 = time.perf_counter()
            lists = {str(i): result.candidates for i, result in enumerate(found)}
            if self.transformer.combine_method == "rrf":
                candidates = reciprocal_rank_fusion(
                    lists, self.transformer.combine_k, label="queries"
                )
            else:
                candidates = round_robin(lists, label="queries")
            trace.timings["combine"] = time.perf_counter() - t0
        trace.candidates = collapse_duplicates(candidates)[:fetch]
        if self.transformer:
            trace.query_lists = [collapse_duplicates(r.candidates)[:fetch] for r in found]
        embed_tokens = sum(result.embed_tokens for result in found)
        trace.usage["embed_tokens"] = embed_tokens
        trace.cost_usd += embedding_cost(self.embedder.model, embed_tokens)
        return trace

    def transform(self, trace: Trace) -> list[Query]:
        """Run the query transformation and record it in the trace. Returns the queries
        the retriever searches with; the ones the reranker scores against are kept in
        the trace for `rerank`."""
        transformer = self.transformer
        result = transformer.transform(trace.question)
        trace.queries = route(trace.question, result.queries, transformer.retrieve_with)
        trace.rerank_queries = route(trace.question, result.queries, transformer.rerank_with)
        trace.transform = {
            "name": transformer.name,
            "retrieve_with": transformer.retrieve_with,
            "rerank_with": transformer.rerank_with,
            "calls": result.calls,
            "cached": result.cached,
            **result.notes,
        }
        if len(trace.queries) > 1:
            trace.transform["combine"] = transformer.combine_method
        trace.models["transformer"] = transformer.model
        trace.timings |= result.timings
        if result.calls:
            trace.usage |= {
                "transform_input_tokens": result.input_tokens,
                "transform_output_tokens": result.output_tokens,
            }
            trace.cost_usd += generation_cost(
                transformer.model, result.input_tokens, result.output_tokens
            )
        return trace.queries

    def rerank(self, trace: Trace, k: int | None = None) -> Trace:
        """Reorder the top `depth` candidates of a first-stage trace and cut to `k`.

        Candidates below the reranker's depth keep their first-stage order underneath
        the reranked ones. So a depth at or below `k` only reorders the top of the list,
        and a depth above `k` can also bring in chunks the first stage ranked below `k`.
        """
        k = k or self.cfg.top_k
        if self.reranker is None:
            trace.candidates = trace.candidates[:k]
            return trace
        depth = self.reranker.depth
        trace.retrieved = trace.candidates
        head = trace.retrieved[:depth]
        # One pass per query the reranker scores against: the question, or with a
        # transformation that says so, each sub-question.
        queries = trace.rerank_queries or [Query(trace.question)]
        results = [self.reranker.rerank(query.text, head) for query in queries]
        if len(results) == 1:
            result = results[0]
        else:
            result = merge(head, results, self.transformer.rerank_combine)
        tail = [
            replace(candidate, rank=len(result.candidates) + i)
            for i, candidate in enumerate(trace.retrieved[depth:], start=1)
        ]
        trace.candidates = (result.candidates + tail)[:k]
        trace.models["reranker"] = self.reranker.model
        trace.timings |= result.timings
        trace.rerank = {"depth": depth, "cached": result.cached, **result.notes}
        if result.input_tokens or result.output_tokens:
            trace.usage |= {
                "rerank_input_tokens": result.input_tokens,
                "rerank_output_tokens": result.output_tokens,
            }
            trace.cost_usd += generation_cost(
                self.reranker.model, result.input_tokens, result.output_tokens
            )
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

    def ask(self, question: str, history: list[Turn] | None = None) -> Trace:
        """Answer one question. `history` is the chat so far as (question, answer)
        pairs; with a follow-up rewriter configured, a message that leans on it is
        first rewritten into a question that stands on its own."""
        standalone = None
        if self.followup is not None and history:
            standalone = self.followup.rewrite(history, question)
        trace = self.retrieve(standalone.question if standalone else question)
        if standalone:
            trace.asked = question
            trace.models["followup"] = self.followup.model
            trace.timings = standalone.timings | trace.timings
            trace.usage |= {
                "followup_input_tokens": standalone.input_tokens,
                "followup_output_tokens": standalone.output_tokens,
            }
            trace.cost_usd += generation_cost(
                self.followup.model, standalone.input_tokens, standalone.output_tokens
            )
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
