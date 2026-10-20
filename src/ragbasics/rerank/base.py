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


def merge(candidates: list[Candidate], results: list[Reranked], method: str) -> Reranked:
    """One ordering from several: the same candidates reranked against each of several
    queries (the sub-questions of a question, Stage 7).

    round_robin  every query's first choice, then every query's second, and so on.
                 Scores are never compared across queries.
    max          a chunk's best score over the queries. One query can score everything
                 higher than another, and then it decides the order.

    Ties and repeats resolve in the order the queries were given.
    """
    if method == "max":
        best: dict[str, float] = {}
        for result in results:
            for c in result.candidates:
                best[c.chunk.chunk_id] = max(c.score, best.get(c.chunk.chunk_id, c.score))
        merged = reorder(candidates, [best[c.chunk.chunk_id] for c in candidates], "rerank")
    elif method == "round_robin":
        # Imported here for the same reason as in `passage_text`.
        from ragbasics.retrieval.fusion import round_robin

        merged = round_robin({str(i): r.candidates for i, r in enumerate(results)}, "rerank")
    else:
        raise ValueError(f"unknown way to merge reranked lists: '{method}'")
    timings: dict[str, float] = {}
    notes: dict[str, Any] = {}
    for result in results:
        for step, seconds in result.timings.items():
            timings[step] = timings.get(step, 0.0) + seconds
        for key, value in result.notes.items():
            # Counts add up, flags stay set, and anything else keeps its last value.
            if isinstance(value, bool):
                notes[key] = notes.get(key, False) or value
            elif isinstance(value, int | float):
                notes[key] = notes.get(key, 0) + value
            else:
                notes[key] = value
    return Reranked(
        candidates=merged,
        timings=timings,
        input_tokens=sum(r.input_tokens for r in results),
        output_tokens=sum(r.output_tokens for r in results),
        cached=all(r.cached for r in results),
        notes=notes | {"queries": len(results), "combine": method},
    )
