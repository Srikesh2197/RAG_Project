"""Exact vector search in NumPy.

Every stored vector is scaled to length 1. The dot product of two unit vectors is their
cosine similarity, so scoring a query against the whole index is one matrix-vector
product. "Exact" means every chunk is scored; nothing is approximated (Stage 11 covers
the approximate kind).
"""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from ragbasics.registry import register
from ragbasics.types import Candidate, Chunk


def unit_rows(matrix: np.ndarray) -> np.ndarray:
    """Scale each row to length 1. All-zero rows stay zero."""
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)


@register("store", "numpy")
class NumpyStore:
    def __init__(self) -> None:
        self.vectors: np.ndarray | None = None  # (number of chunks, dimensions), unit rows
        self.chunks: list[Chunk] = []

    def __len__(self) -> int:
        return len(self.chunks)

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        vectors = unit_rows(vectors)
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        if self.vectors is None:
            self.vectors = vectors
        elif vectors.shape[1] != self.vectors.shape[1]:
            raise ValueError(
                f"index holds {self.vectors.shape[1]}-dimensional vectors; "
                f"got {vectors.shape[1]}"
            )
        else:
            self.vectors = np.vstack([self.vectors, vectors])
        self.chunks += chunks

    def search(self, query_vector: np.ndarray, k: int) -> list[Candidate]:
        """The `k` chunks with the highest cosine similarity to the query, best first.

        Equal scores keep insertion order, so results are deterministic.
        """
        if self.vectors is None:
            return []
        query = unit_rows(np.asarray(query_vector).reshape(1, -1))[0]
        if query.shape[0] != self.vectors.shape[1]:
            raise ValueError(
                f"query has {query.shape[0]} dimensions; index has {self.vectors.shape[1]}"
            )
        scores = self.vectors @ query
        order = np.argsort(-scores, kind="stable")[:k]
        return [
            Candidate(chunk=self.chunks[i], score=float(scores[i]), rank=rank)
            for rank, i in enumerate(order, start=1)
        ]

    def save(self, directory: Path, meta: dict[str, Any]) -> None:
        """Write vectors.npy, chunks.jsonl and meta.json (what built this index)."""
        directory.mkdir(parents=True, exist_ok=True)
        vectors = self.vectors if self.vectors is not None else np.zeros((0, 0), np.float32)
        np.save(directory / "vectors.npy", vectors)
        with open(directory / "chunks.jsonl", "w") as f:
            for chunk in self.chunks:
                f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
        (directory / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    @classmethod
    def load(cls, directory: Path) -> tuple["NumpyStore", dict[str, Any]]:
        store = cls()
        with open(directory / "chunks.jsonl") as f:
            store.chunks = [Chunk(**json.loads(line)) for line in f]
        if store.chunks:
            store.vectors = np.load(directory / "vectors.npy")
        return store, json.loads((directory / "meta.json").read_text())
