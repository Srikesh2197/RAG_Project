"""Semantic chunking: cut where the topic changes, as measured by an embedding model.

    1. Split the article into sentences.
    2. Embed each sentence together with its neighbours (`window` on each side), which
       smooths out one-off sentences.
    3. Take the cosine distance between each sentence's vector and the next one's.
    4. Cut at the largest distances: those above the article's `percentile`.
    5. A segment above `max_size` tokens is packed down by sentence, like the
       recursive chunker.

Chunk size is an outcome, not a setting: `percentile` 90 cuts at one sentence gap in
ten. The price is one extra embedding call per sentence at index time, on top of
embedding the chunks themselves. Sentence vectors are cached on disk, so trying another
percentile is free.
"""

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from ragbasics.chunking import units
from ragbasics.chunking.recursive import SentenceChunker
from ragbasics.chunking.units import Span
from ragbasics.index.numpy_store import unit_rows
from ragbasics.registry import build, register
from ragbasics.types import Chunk, Document

EMBED_GROUP = 40  # articles embedded per call, so an interrupted run keeps its progress


def breakpoints(vectors: np.ndarray, percentile: float) -> list[int]:
    """Indexes i where a cut falls between sentence i and sentence i + 1."""
    if len(vectors) < 2:
        return []
    rows = unit_rows(vectors)
    distances = 1.0 - np.sum(rows[:-1] * rows[1:], axis=1)
    threshold = np.percentile(distances, percentile)
    return [i for i, d in enumerate(distances) if d > threshold]


@register("chunker", "semantic")
class SemanticChunker:
    def __init__(
        self,
        embedder: dict[str, Any],
        percentile: float = 90,
        max_size: int = 512,
        window: int = 1,
        cache_dir: str | None = None,
    ):
        self.embedder = build("embedder", embedder["name"], **embedder.get("params", {}))
        self.percentile = percentile
        self.window = window
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.packer = SentenceChunker(size=max_size)
        self.billed_tokens = 0  # embedding tokens paid for by this chunker, for the ledger
        self.model = self.embedder.model
        self._memory: dict[str, np.ndarray] = {}

    # --- Sentence vectors ---------------------------------------------------------------

    def _windows(self, document: Document) -> tuple[list[Span], list[str]]:
        """Sentence spans, and for each the text embedded for it: the sentence with
        `window` neighbours on each side."""
        spans = [
            s
            for s in units.sentences(document.text, 0, len(document.text))
            if document.text[s[0] : s[1]].strip()
        ]
        texts = [
            document.text[spans[max(0, i - self.window)][0] : spans[
                min(len(spans) - 1, i + self.window)
            ][1]]
            for i in range(len(spans))
        ]
        return spans, texts

    def _key(self, texts: list[str]) -> str:
        digest = hashlib.sha256(self.model.encode())
        for text in texts:
            digest.update(b"\x00" + text.encode())
        return digest.hexdigest()[:32]

    def _cached(self, key: str) -> np.ndarray | None:
        if key in self._memory:
            return self._memory[key]
        if self.cache_dir and (self.cache_dir / f"{key}.npy").exists():
            self._memory[key] = np.load(self.cache_dir / f"{key}.npy")
            return self._memory[key]
        return None

    def _store(self, key: str, vectors: np.ndarray) -> None:
        self._memory[key] = vectors
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            np.save(self.cache_dir / f"{key}.npy", vectors)

    def _pending(self, documents: list[Document]) -> list[tuple[str, list[str]]]:
        """(cache key, texts) for each article whose sentence vectors are not cached."""
        pending = []
        for document in documents:
            _, texts = self._windows(document)
            if len(texts) > 1 and self._cached(self._key(texts)) is None:
                pending.append((self._key(texts), texts))
        return pending

    def texts_to_embed(self, documents: list[Document]) -> list[str]:
        """What `prepare` would send to the embedder. Used to print the cost first."""
        return [text for _, texts in self._pending(documents) for text in texts]

    def prepare(self, documents: list[Document]) -> None:
        """Embed the sentences of every article not yet cached, a group of articles per call."""
        pending = self._pending(documents)
        for i in range(0, len(pending), EMBED_GROUP):
            group = pending[i : i + EMBED_GROUP]
            embedded = self.embedder.embed([text for _, texts in group for text in texts])
            self.billed_tokens += embedded.tokens
            row = 0
            for key, texts in group:
                self._store(key, embedded.vectors[row : row + len(texts)])
                row += len(texts)

    # --- Chunking -----------------------------------------------------------------------

    def segments(self, document: Document) -> list[Span]:
        """The article cut at its semantic breakpoints, before the size cap."""
        spans, texts = self._windows(document)
        if not spans:
            return []
        cuts: list[int] = []
        if len(spans) > 1:
            if self._cached(self._key(texts)) is None:
                self.prepare([document])
            cuts = breakpoints(self._cached(self._key(texts)), self.percentile)
        # Sentence spans can skip leading whitespace-only pieces; chunks still tile the text.
        bounds = [0, *(spans[i][1] for i in cuts), len(document.text)]
        return list(zip(bounds, bounds[1:], strict=False))

    def chunk(self, document: Document) -> list[Chunk]:
        spans: list[Span] = []
        for start, end in self.segments(document):
            spans += self.packer.split(document.text, start, end)
        return units.to_chunks(document, spans)
