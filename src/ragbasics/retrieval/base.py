"""What every retriever takes and returns.

A retriever turns a question into a ranked list of chunks. It is given the pipeline's
index at call time: the vector store, the embedder that built it, and the directory it
is saved in. Retrievers add no chunks of their own, so dense, sparse and hybrid search
all rank the same chunks and differ only in how.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ragbasics.types import Candidate


class Index(Protocol):
    """The parts of a pipeline a retriever reads."""

    embedder: Any
    store: Any
    index_dir: Path


@dataclass
class Retrieval:
    candidates: list[Candidate]  # the final ranked list
    # Each underlying retriever's own list. One entry for dense or sparse search; for
    # hybrid, the lists that were fused.
    lists: dict[str, list[Candidate]]
    timings: dict[str, float] = field(default_factory=dict)  # seconds per step
    embed_tokens: int = 0


class Retriever(Protocol):
    def retrieve(self, question: str, k: int, index: Index) -> Retrieval: ...
