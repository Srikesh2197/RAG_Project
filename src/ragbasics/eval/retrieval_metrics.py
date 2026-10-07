"""Retrieval metrics, written by hand.

Everything is defined against gold evidence spans: a chunk is relevant if it overlaps
at least one gold span of the question. `chunks` is always the ranked list, best first.
"""

import math
from typing import Any

from ragbasics.types import Chunk, EvidenceSpan

Range = tuple[str, int, int]  # (doc_id, start_char, end_char), end exclusive


def overlaps(chunk: Chunk, span: EvidenceSpan) -> bool:
    return (
        chunk.doc_id == span.doc_id
        and chunk.start_char < span.end_char
        and span.start_char < chunk.end_char
    )


def relevance(chunks: list[Chunk], spans: tuple[EvidenceSpan, ...]) -> list[bool]:
    """For each ranked chunk, whether it overlaps any gold span."""
    return [any(overlaps(chunk, span) for span in spans) for chunk in chunks]


def gold_ranks(chunks: list[Chunk], spans: tuple[EvidenceSpan, ...]) -> list[int | None]:
    """For each gold span, the rank (1 is best) of the first chunk that overlaps it."""
    return [
        next((rank for rank, c in enumerate(chunks, start=1) if overlaps(c, span)), None)
        for span in spans
    ]


def evidence_recall(chunks: list[Chunk], spans: tuple[EvidenceSpan, ...], k: int) -> float:
    """Share of the question's gold spans that the top `k` chunks touch."""
    ranks = gold_ranks(chunks[:k], spans)
    return sum(rank is not None for rank in ranks) / len(spans)


def full_support(chunks: list[Chunk], spans: tuple[EvidenceSpan, ...], k: int) -> float:
    """1 if the top `k` chunks touch every gold span, else 0. A multi-hop answer needs all."""
    return float(evidence_recall(chunks, spans, k) == 1.0)


def reciprocal_rank(relevant: list[bool], k: int = 10) -> float:
    """1 / rank of the first relevant chunk in the top `k`; 0 if there is none."""
    for rank, hit in enumerate(relevant[:k], start=1):
        if hit:
            return 1.0 / rank
    return 0.0


def ndcg(relevant: list[bool], total_relevant: int, k: int = 10) -> float:
    """Normalised discounted cumulative gain at `k`, with gain 1 per relevant chunk.

    A relevant chunk at rank r adds 1 / log2(r + 1), so rank 1 adds 1 and rank 10 adds
    0.29. The sum is divided by the best possible sum: all `total_relevant` chunks of
    the index placed at the top.
    """
    dcg = sum(1.0 / math.log2(rank + 1) for rank, hit in enumerate(relevant[:k], start=1) if hit)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(total_relevant, k) + 1))
    return dcg / ideal if ideal else 0.0


# --- A fixed token budget instead of a fixed number of chunks ---------------------------
#
# Five 1,024-token chunks hold eight times the text of five 128-token chunks, so
# recall@5 rewards large chunks for nothing. Reading the ranked list until a fixed number
# of tokens is used gives every chunker the same amount of text.


def within_budget(chunks: list[Chunk], budget: int, encoding: Any) -> list[Range]:
    """The character ranges covered by the first `budget` tokens of the ranked chunks.

    Chunks are taken whole in rank order. The one that crosses the budget is cut at the
    budget, so every chunk size gets exactly the same number of tokens.
    """
    kept: list[Range] = []
    used = 0
    for chunk in chunks:
        tokens = encoding.encode(chunk.text, disallowed_special=())
        room = budget - used
        if room <= 0:
            break
        if len(tokens) <= room:
            kept.append((chunk.doc_id, chunk.start_char, chunk.end_char))
            used += len(tokens)
        else:
            prefix = encoding.decode(tokens[:room])
            kept.append((chunk.doc_id, chunk.start_char, chunk.start_char + len(prefix)))
            break
    return kept


def _covered_chars(span: EvidenceSpan, ranges: list[Range]) -> int:
    """Characters of `span` inside the union of `ranges` (overlapping ranges count once)."""
    pieces = sorted(
        (max(start, span.start_char), min(end, span.end_char))
        for doc_id, start, end in ranges
        if doc_id == span.doc_id and start < span.end_char and span.start_char < end
    )
    covered, reach = 0, span.start_char
    for start, end in pieces:
        start = max(start, reach)
        if end > start:
            covered += end - start
            reach = end
    return covered


def budget_recall(ranges: list[Range], spans: tuple[EvidenceSpan, ...]) -> float:
    """Share of gold spans touched by the text inside the budget."""
    return sum(_covered_chars(span, ranges) > 0 for span in spans) / len(spans)


def budget_precision(ranges: list[Range], spans: tuple[EvidenceSpan, ...]) -> float:
    """Share of the retrieved text inside the budget that is gold evidence.

    Measured in characters, which track tokens closely and need no second tokenisation.
    Text retrieved twice (overlapping chunks) is counted twice in the denominator, since
    it used the budget twice, and once in the numerator.
    """
    retrieved = sum(end - start for _, start, end in ranges)
    gold = sum(_covered_chars(span, ranges) for span in spans)
    return gold / retrieved if retrieved else 0.0
