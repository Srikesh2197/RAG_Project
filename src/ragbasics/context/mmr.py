"""Maximal marginal relevance (Carbonell and Goldstein, 1998), by hand.

Pick chunks one at a time. The first is the most relevant. Each next one maximises

    lam * relevance(chunk) - (1 - lam) * max similarity(chunk, any chunk already picked)

so a chunk that repeats one already picked is pushed down even if it is relevant.
`lam = 1` is plain relevance order; `lam = 0` ignores relevance after the first pick and
takes whatever is least like what was picked.

What it trades away: a second chunk that says nearly the same thing as the first. That
is the right trade for repeated boilerplate and syndicated copies, and the wrong one when
a multi-hop question needs two similar-looking chunks (two articles about the same
company, one per fact).
"""

import numpy as np


def mmr(relevance: np.ndarray, similarity: np.ndarray, k: int, lam: float) -> list[int]:
    """Indices of the `k` items MMR picks, in the order it picks them.

    relevance   (n,) how relevant each item is to the question; any scale, but the
                similarity term is a cosine, so relevance should be on a comparable
                [0, 1]-like scale (see `min_max`)
    similarity  (n, n) cosine similarity between items
    Ties go to the lower index, so the result is deterministic.
    """
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"lam must be in [0, 1], got {lam}")
    n = len(relevance)
    if similarity.shape != (n, n):
        raise ValueError(f"similarity is {similarity.shape}, expected ({n}, {n})")
    if n == 0 or k <= 0:
        return []
    picked = [int(np.argmax(relevance))]
    # Highest similarity of every item to anything picked so far.
    redundancy = similarity[:, picked[0]].astype(float).copy()
    while len(picked) < min(k, n):
        scores = lam * relevance - (1.0 - lam) * redundancy
        scores[picked] = -np.inf
        best = int(np.argmax(scores))
        picked.append(best)
        redundancy = np.maximum(redundancy, similarity[:, best])
    return picked


def min_max(scores: np.ndarray) -> np.ndarray:
    """Scale to [0, 1]. A reranker's scores (a logit, a probability, a fused rank score)
    are on its own scale; MMR weighs them against a cosine, so both need one range.
    All-equal scores become all ones."""
    scores = np.asarray(scores, dtype=float)
    if len(scores) == 0:
        return scores
    low, high = scores.min(), scores.max()
    if high == low:
        return np.ones_like(scores)
    return (scores - low) / (high - low)
