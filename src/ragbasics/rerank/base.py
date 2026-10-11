"""What every reranker takes and returns.

A reranker is the second stage of two-stage retrieval. The first stage (dense, sparse or
hybrid search) is cheap and picks candidates from the whole index. The reranker is
expensive, reads the question together with each candidate, and reorders them. It adds
no chunks: whatever the first stage missed stays missed.

`depth` is how many of the first stage's candidates the reranker reads. Chunks below
that keep their first-stage order underneath the reranked ones.
"""

from dataclasses import dataclass, field
from typing import Any, Protocol

from ragbasics.types import Candidate, Chunk


@dataclass
class Reranked:
    candidates: list[Candidate]  # the same chunks, reordered, with the reranker's scores
    timings: dict[str, float] = field(default_factory=dict)  # seconds per step
    # Tokens billed at list price for this call (an LLM reranker); 0 for a local model.
    input_tokens: int = 0
    output_tokens: int = 0
    cached: bool = False  # every score or the whole ordering came from the disk cache
    # Anything worth showing in the trace: pairs truncated, a repaired LLM ordering.
    notes: dict[str, Any] = field(default_factory=dict)


class Reranker(Protocol):
    model: str
    depth: int

    def rerank(self, question: str, candidates: list[Candidate]) -> Reranked: ...


def passage_text(chunk: Chunk, header: bool) -> str:
    """What the reranker reads for a chunk: its text, optionally under a title, source
    and date line. The text sent to the generator is not changed by this."""
    if not header:
        return chunk.text
    # Imported here so that the sparse retriever's module is not a dependency of every
    # reranker import.
    from ragbasics.retrieval.sparse import header as header_line

    line = header_line(chunk.metadata)
    return f"{line}\n{chunk.text}" if line else chunk.text


def reorder(candidates: list[Candidate], scores: list[float], name: str) -> list[Candidate]:
    """Candidates sorted by `scores`, best first, renumbered from rank 1.

    Equal scores keep the first stage's order, so the result is deterministic and a
    reranker that says nothing (all scores equal) changes nothing.
    """
    order = sorted(range(len(candidates)), key=lambda i: (-scores[i], candidates[i].rank))
    return [
        Candidate(chunk=candidates[i].chunk, score=scores[i], rank=rank, retriever=name)
        for rank, i in enumerate(order, start=1)
    ]
