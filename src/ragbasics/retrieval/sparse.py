"""Sparse retrieval: BM25 over the terms of each chunk.

"Sparse" because a chunk is described by the few terms it contains out of the whole
vocabulary, where an embedding has a number in every dimension.

The BM25 index is built from the chunks already in the vector store, so it needs no
embedding call. It is saved under the index directory, in `bm25/<settings>/`, one
directory per tokeniser setting, and rebuilt when the chunks change. The dense index's
own files are not touched.
"""

import hashlib
import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from ragbasics.index.bm25 import BM25Index
from ragbasics.registry import register
from ragbasics.retrieval.base import Index, Retrieval
from ragbasics.retrieval.tokenize import Tokenizer
from ragbasics.types import Candidate, Chunk


def header(metadata: dict[str, Any]) -> str:
    """'Title. TechCrunch, October 7, 2023.' from whichever of those the chunk carries.

    The date is written out because that is how a question writes it; "2023-10-07"
    shares only the year with "October 7, 2023".
    """
    parts = [str(metadata["title"])] if metadata.get("title") else []
    origin = [str(metadata["source"])] if metadata.get("source") else []
    if metadata.get("published_at"):
        try:
            date = datetime.fromisoformat(str(metadata["published_at"]))
            origin.append(f"{date:%B} {date.day}, {date.year}")
        except ValueError:
            pass
    if origin:
        parts.append(", ".join(origin))
    return ". ".join(parts) + "." if parts else ""


@register("retriever", "sparse")
class SparseRetriever:
    def __init__(
        self,
        k1: float = 1.2,
        b: float = 0.75,
        stopwords: bool = False,
        stemmer: str = "none",
        header: bool = False,
    ):
        self.k1 = k1
        self.b = b
        self.tokenizer = Tokenizer(stopwords=stopwords, stemmer=stemmer)
        # Also index each chunk's title, source and date. The text returned and sent to
        # the generator stays the chunk's own.
        self.header = header
        self._lock = threading.Lock()
        self._built: tuple[int, int, BM25Index] | None = None  # (store, chunk count, index)

    def settings(self) -> dict[str, Any]:
        """What the saved index depends on. `k1` and `b` are not in it: they are applied
        at query time to the stored counts."""
        return {**self.tokenizer.settings(), "header": self.header}

    def indexed_text(self, chunk: Chunk) -> str:
        text = chunk.text_to_embed
        return f"{header(chunk.metadata)}\n{text}" if self.header else text

    def directory(self, index_dir: Path) -> Path:
        key = hashlib.sha256(json.dumps(self.settings(), sort_keys=True).encode()).hexdigest()
        return index_dir / "bm25" / key[:12]

    def index_for(self, index: Index) -> BM25Index:
        """The BM25 index over the store's chunks: from memory, from disk, or built now."""
        chunks = index.store.chunks
        with self._lock:  # the evaluator retrieves from several threads
            if self._built and self._built[:2] == (id(index.store), len(chunks)):
                return self._built[2]
            digest = hashlib.sha256("\n".join(c.chunk_id for c in chunks).encode()).hexdigest()
            meta = {**self.settings(), "chunks": len(chunks), "chunk_ids": digest}
            directory = self.directory(index.index_dir)
            bm25 = None
            if (directory / "meta.json").exists():
                bm25, saved = BM25Index.load(directory)
                if saved != meta:
                    bm25 = None  # chunks were added since it was saved
            if bm25 is None:
                bm25 = BM25Index.build([self.tokenizer(self.indexed_text(c)) for c in chunks])
                bm25.save(directory, meta)
            self._built = (id(index.store), len(chunks), bm25)
            return bm25

    def retrieve(self, question: str, k: int, index: Index) -> Retrieval:
        bm25 = self.index_for(index)
        t0 = time.perf_counter()
        hits = bm25.search(self.tokenizer(question), k, self.k1, self.b)
        chunks = index.store.chunks
        found = [
            Candidate(chunk=chunks[i], score=score, rank=rank, retriever="sparse")
            for rank, (i, score) in enumerate(hits, start=1)
        ]
        return Retrieval(
            candidates=found,
            lists={"sparse": found},
            timings={"sparse_search": time.perf_counter() - t0},
        )
