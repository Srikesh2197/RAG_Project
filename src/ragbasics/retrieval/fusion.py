"""Combining ranked lists from different retrievers.

A cosine similarity and a BM25 score are on unrelated scales. A cosine here sits between
about 0.2 and 0.7; a BM25 score has no upper limit and grows with the number of terms in
the question. Adding them would let BM25 decide everything. There are two ways round it:

- Reciprocal rank fusion throws the scores away and uses only each chunk's rank.
- Weighted fusion first rescales each list's scores to a common range, then mixes them
  with a weight. It keeps the size of a lead, which ranks lose, and the weight has to be
  tuned on labelled questions.
"""

import statistics

from ragbasics.types import Candidate

NORMALISERS = ("minmax", "zscore")


def _ordered(scores: dict[str, float], lists: dict[str, list[Candidate]]) -> list[Candidate]:
    """Candidates sorted by fused score. Equal scores keep the order chunks were first
    seen in (the first list, then the second), so results are deterministic."""
    chunks = {}
    for candidates in lists.values():
        for candidate in candidates:
            chunks.setdefault(candidate.chunk.chunk_id, candidate.chunk)
    order = sorted(chunks, key=lambda chunk_id: -scores[chunk_id])
    return [
        Candidate(chunk=chunks[chunk_id], score=scores[chunk_id], rank=rank, retriever="hybrid")
        for rank, chunk_id in enumerate(order, start=1)
    ]


def reciprocal_rank_fusion(
    lists: dict[str, list[Candidate]], k: float = 60, weights: dict[str, float] | None = None
) -> list[Candidate]:
    """Each list gives a chunk 1 / (k + rank); a chunk's fused score is the sum.

    A chunk missing from a list gets nothing from it. `k` sets how much the top ranks
    dominate: at k = 1, rank 1 is worth five times rank 9; at k = 60, 1.13 times. A
    large k therefore favours chunks that both lists found over a chunk one list put
    first.
    """
    scores: dict[str, float] = {}
    for name, candidates in lists.items():
        weight = 1.0 if weights is None else weights[name]
        for candidate in candidates:
            chunk_id = candidate.chunk.chunk_id
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (k + candidate.rank)
    return _ordered(scores, lists)


def normalise(scores: list[float], method: str) -> list[float]:
    """Rescale one list's scores so lists from different retrievers can be mixed.

    minmax  (score - lowest) / (highest - lowest): the best is 1 and the worst is 0.
            One outlying top score squeezes everything else towards 0.
    zscore  (score - mean) / standard deviation: how far above the list's average.
            Less sensitive to a single outlier.

    Both depend on which scores are in the list, so on how deep the retriever searched.
    """
    if method not in NORMALISERS:
        raise ValueError(f"unknown normaliser '{method}'; choose from {NORMALISERS}")
    if not scores:
        return []
    if method == "minmax":
        low, spread = min(scores), max(scores) - min(scores)
    else:
        low, spread = statistics.fmean(scores), statistics.pstdev(scores)
    if spread == 0:
        return [0.0] * len(scores)  # every score equal: the list says nothing
    return [(score - low) / spread for score in scores]


def weighted_fusion(
    lists: dict[str, list[Candidate]], weights: dict[str, float], method: str = "minmax"
) -> list[Candidate]:
    """Fused score = sum over lists of weight * normalised score.

    A chunk missing from a list gets that list's lowest normalised score: the list
    searched and ranked it below everything it returned.
    """
    normalised: dict[str, dict[str, float]] = {}
    floor: dict[str, float] = {}
    for name, candidates in lists.items():
        values = normalise([candidate.score for candidate in candidates], method)
        normalised[name] = {c.chunk.chunk_id: v for c, v in zip(candidates, values, strict=True)}
        floor[name] = min(values, default=0.0)
    chunk_ids = {chunk_id for values in normalised.values() for chunk_id in values}
    scores = {
        chunk_id: sum(
            weights[name] * normalised[name].get(chunk_id, floor[name]) for name in lists
        )
        for chunk_id in chunk_ids
    }
    return _ordered(scores, lists)

