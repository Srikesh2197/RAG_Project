import numpy as np
import pytest

from ragbasics.eval.stats import bootstrap_ci, cohen_kappa, paired_bootstrap


def ids(n):
    return [f"q{i}" for i in range(n)]


def test_interval_contains_the_mean_and_is_repeatable():
    values = [1.0] * 60 + [0.0] * 40
    first = bootstrap_ci(values, ids(100), seed=3)
    assert first.mean == 0.6 and first.n == 100
    assert first.low < 0.6 < first.high
    assert first == bootstrap_ci(values, ids(100), seed=3)


def test_question_level_interval_matches_the_textbook_formula():
    # For a share p over n independent questions the standard error is sqrt(p(1-p)/n).
    values = [1.0] * 300 + [0.0] * 200
    interval = bootstrap_ci(values, ids(500), samples=4000)
    half_width = 1.96 * np.sqrt(0.6 * 0.4 / 500)
    assert interval.high - interval.low == pytest.approx(2 * half_width, rel=0.1)


def test_identical_values_give_a_zero_width_interval():
    interval = bootstrap_ci([1.0] * 20, ids(20))
    assert (interval.low, interval.high) == (1.0, 1.0)


def test_questions_that_move_together_widen_the_interval():
    # 100 questions in 10 clusters; inside a cluster every question has the same result.
    values = [float(c < 6) for c in range(10) for _ in range(10)]
    clusters = [f"c{c}" for c in range(10) for _ in range(10)]
    by_cluster = bootstrap_ci(values, clusters)
    by_question = bootstrap_ci(values, ids(100))
    assert by_cluster.mean == by_question.mean == 0.6
    # The data hold 10 independent results, not 100.
    assert (by_cluster.high - by_cluster.low) > 2.5 * (by_question.high - by_question.low)


def test_one_cluster_holds_no_information_about_spread():
    interval = bootstrap_ci([1.0, 0.0, 1.0, 1.0], ["c"] * 4)
    assert (interval.low, interval.high) == (0.75, 0.75)


def test_pairing_removes_question_difficulty_from_the_difference():
    rng = np.random.default_rng(0)
    b = (rng.random(300) < 0.5).astype(float)
    a = b.copy()
    a[np.flatnonzero(b == 0)[:15]] = 1.0  # system a fixes 15 of b's failures, breaks nothing
    paired = paired_bootstrap(a.tolist(), b.tolist(), ids(300))
    assert paired.mean == pytest.approx(0.05)
    assert paired.low > 0  # a real difference: it never goes the other way
    # The two separate intervals overlap heavily, which would hide the same difference.
    assert bootstrap_ci(a.tolist(), ids(300)).low < bootstrap_ci(b.tolist(), ids(300)).high


def test_paired_needs_the_same_questions():
    with pytest.raises(ValueError):
        paired_bootstrap([1.0], [1.0, 0.0], ["a", "b"])


def test_no_values_gives_an_empty_interval():
    assert bootstrap_ci([], []).n == 0


def test_kappa_discounts_agreement_that_chance_would_give():
    yes, no = True, False
    assert cohen_kappa([yes, no, yes, no], [yes, no, yes, no]) == 1.0
    # 9 of 10 agree, but both raters say yes 90% of the time: chance alone gives 82%.
    a = [yes] * 9 + [no]
    b = [yes] * 8 + [no, yes]
    assert sum(x == y for x, y in zip(a, b, strict=True)) == 8
    assert cohen_kappa(a, b) == pytest.approx((0.8 - 0.82) / (1 - 0.82))
