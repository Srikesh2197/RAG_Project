"""Confidence intervals by bootstrap.

A bootstrap asks: if we had drawn a different set of questions, how much would the
metric move? It answers by redrawing from the questions we have, with replacement,
many times, and reading the spread of the metric over those redraws.

Questions in this dataset share gold evidence, so their results move together. The
unit that is redrawn is therefore the evidence cluster, not the question. Redrawing
single questions would treat 500 correlated results as 500 independent ones and give
an interval that is too narrow.
"""

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class Interval:
    mean: float
    low: float
    high: float
    n: int  # questions the mean is taken over

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def bootstrap_ci(
    values: list[float],
    clusters: list[str],
    samples: int = 2000,
    seed: int = 0,
    level: float = 0.95,
) -> Interval:
    """Mean of `values` with a percentile interval from redrawing whole clusters.

    `values[i]` is one question's score and `clusters[i]` its cluster id. Each redraw
    picks as many clusters as there are, with replacement, and takes the mean over
    every question in the picked clusters. Pass a distinct id per question to get the
    ordinary question-level bootstrap.
    """
    if len(values) == 0:
        return Interval(float("nan"), float("nan"), float("nan"), 0)
    scores = np.asarray(values, dtype=float)
    _, cluster_index = np.unique(np.asarray(clusters), return_inverse=True)
    # A redraw only needs each cluster's total and size.
    sums = np.bincount(cluster_index, weights=scores)
    counts = np.bincount(cluster_index)

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(sums), size=(samples, len(sums)))
    means = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    tail = (1 - level) / 2
    low, high = np.quantile(means, [tail, 1 - tail])
    return Interval(float(scores.mean()), float(low), float(high), len(scores))


def paired_bootstrap(
    a: list[float],
    b: list[float],
    clusters: list[str],
    samples: int = 2000,
    seed: int = 0,
    level: float = 0.95,
) -> Interval:
    """Interval for mean(a) - mean(b) when both were scored on the same questions.

    Each redraw uses the same questions for both systems, so a hard question lowers
    both and cancels out of the difference. That is the same as bootstrapping the
    per-question differences. If the interval excludes 0, the difference is unlikely
    to be an accident of which questions were drawn.
    """
    if len(a) != len(b):
        raise ValueError(f"paired comparison needs equal lengths; got {len(a)} and {len(b)}")
    differences = (np.asarray(a, dtype=float) - np.asarray(b, dtype=float)).tolist()
    return bootstrap_ci(differences, clusters, samples, seed, level)


def cohen_kappa(a: list[bool], b: list[bool]) -> float:
    """Agreement between two raters beyond what their yes-rates give by chance.

    Two raters who both say yes 90% of the time agree on 82% of items without looking
    at them. Kappa rescales agreement so that this chance level is 0 and perfect
    agreement is 1.
    """
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    yes_a, yes_b = sum(a) / n, sum(b) / n
    chance = yes_a * yes_b + (1 - yes_a) * (1 - yes_b)
    return (observed - chance) / (1 - chance) if chance < 1 else 1.0
