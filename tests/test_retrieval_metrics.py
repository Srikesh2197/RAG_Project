import math
import random

import pytest

from ragbasics.eval import retrieval_metrics as rm
from ragbasics.types import Chunk, EvidenceSpan


def chunk(doc: str, start: int, end: int) -> Chunk:
    return Chunk(f"{doc}:{start}", doc, "x" * (end - start), start, end)


class CharEncoding:
    """One token per character, so budgets in the tests can be counted by eye."""

    def encode(self, text, disallowed_special=()):
        return list(text)

    def decode(self, tokens):
        return "".join(tokens)


# Two gold sentences in two articles. The ranked list finds the first at rank 2 and the
# second at rank 4; ranks 1 and 3 are the right article but the wrong part, or unrelated.
SPANS = (EvidenceSpan("a", 120, 160), EvidenceSpan("b", 10, 50))
RANKED = [chunk("a", 0, 100), chunk("a", 100, 200), chunk("c", 0, 100), chunk("b", 0, 100)]


def test_a_chunk_is_relevant_only_if_it_overlaps_a_span_in_the_same_document():
    assert rm.relevance(RANKED, SPANS) == [False, True, False, True]
    # Touching end to start is not an overlap: ranges are half-open.
    assert not rm.overlaps(chunk("a", 0, 120), SPANS[0])
    assert rm.overlaps(chunk("a", 0, 121), SPANS[0])
    assert not rm.overlaps(chunk("b", 120, 160), SPANS[0])


def test_recall_counts_spans_and_full_support_needs_all_of_them():
    assert rm.gold_ranks(RANKED, SPANS) == [2, 4]
    assert rm.evidence_recall(RANKED, SPANS, k=1) == 0.0
    assert rm.evidence_recall(RANKED, SPANS, k=2) == 0.5
    assert rm.evidence_recall(RANKED, SPANS, k=4) == 1.0
    assert rm.full_support(RANKED, SPANS, k=3) == 0.0
    assert rm.full_support(RANKED, SPANS, k=4) == 1.0


def test_a_span_cut_by_a_chunk_boundary_is_found_by_either_half():
    span = (EvidenceSpan("a", 90, 110),)
    assert rm.evidence_recall([chunk("a", 0, 100)], span, k=1) == 1.0
    assert rm.evidence_recall([chunk("a", 100, 200)], span, k=1) == 1.0


def test_reciprocal_rank_is_one_over_the_first_relevant_rank():
    assert rm.reciprocal_rank([False, True, False, True]) == 0.5
    assert rm.reciprocal_rank([False] * 10 + [True]) == 0.0  # rank 11 is outside the top 10
    assert rm.reciprocal_rank([]) == 0.0


def test_mrr_and_full_support_can_disagree():
    # One of two gold sentences at rank 1, the other never retrieved.
    ranked = [chunk("a", 100, 200), chunk("c", 0, 100)]
    assert rm.reciprocal_rank(rm.relevance(ranked, SPANS)) == 1.0
    assert rm.full_support(ranked, SPANS, k=2) == 0.0


def test_ndcg_against_a_hand_computed_value():
    relevant = [False, True, False, True]
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    ideal = 1 / math.log2(2) + 1 / math.log2(3)
    assert rm.ndcg(relevant, total_relevant=2) == pytest.approx(dcg / ideal)
    assert rm.ndcg([True, True], total_relevant=2) == 1.0
    assert rm.ndcg([False, False], total_relevant=2) == 0.0
    assert rm.ndcg([False], total_relevant=0) == 0.0


def test_the_token_budget_cuts_the_chunk_that_crosses_it():
    # 100-character chunks, one token per character, budget 250: two whole chunks and
    # the first 50 characters of the third.
    kept = rm.within_budget(RANKED, 250, CharEncoding())
    assert kept == [("a", 0, 100), ("a", 100, 200), ("c", 0, 50)]
    assert rm.within_budget(RANKED, 200, CharEncoding()) == [("a", 0, 100), ("a", 100, 200)]
    assert rm.within_budget(RANKED, 10_000, CharEncoding())[-1] == ("b", 0, 100)


def test_budget_recall_and_precision():
    kept = rm.within_budget(RANKED, 250, CharEncoding())
    # Only the first gold sentence (40 characters) is inside the 250 retrieved characters.
    assert rm.budget_recall(kept, SPANS) == 0.5
    assert rm.budget_precision(kept, SPANS) == pytest.approx(40 / 250)
    assert rm.budget_precision([], SPANS) == 0.0


def test_the_cut_can_remove_gold_text_from_the_last_chunk():
    span = (EvidenceSpan("a", 60, 80),)
    assert rm.budget_recall(rm.within_budget([chunk("a", 0, 100)], 50, CharEncoding()), span) == 0.0
    assert rm.budget_recall(rm.within_budget([chunk("a", 0, 100)], 70, CharEncoding()), span) == 1.0


def test_overlapping_chunks_pay_for_repeated_text_but_gold_counts_once():
    span = (EvidenceSpan("a", 40, 60),)
    kept = [("a", 0, 100), ("a", 50, 150)]  # 50 characters retrieved twice
    assert rm.budget_precision(kept, span) == pytest.approx(20 / 200)


def test_same_budget_gives_small_and_large_chunks_the_same_amount_of_text():
    small = [chunk("a", i, i + 25) for i in range(0, 400, 25)]
    large = [chunk("a", i, i + 200) for i in range(0, 400, 200)]
    for ranked in (small, large):
        kept = rm.within_budget(ranked, 150, CharEncoding())
        assert sum(end - start for _, start, end in kept) == 150


def test_hand_written_metrics_match_ranx():
    """Library equivalent. ranx scores chunk ids, so the fixture puts each gold span in
    its own single-chunk document: chunk-level and span-level recall are then the same."""
    ranx = pytest.importorskip("ranx")
    rng = random.Random(0)
    docs = [f"d{i:02d}" for i in range(40)]
    qrels, run, mine = {}, {}, {"mrr@10": [], "ndcg@10": [], "recall@5": []}
    for q in range(30):
        gold_docs = rng.sample(docs, rng.randint(1, 4))
        spans = tuple(EvidenceSpan(d, 10, 20) for d in gold_docs)
        ranked_docs = rng.sample(docs, 20)
        ranked = [chunk(d, 0, 100) for d in ranked_docs]
        qrels[f"q{q}"] = {f"{d}:0": 1 for d in gold_docs}
        run[f"q{q}"] = {c.chunk_id: 1.0 - i / 100 for i, c in enumerate(ranked)}
        relevant = rm.relevance(ranked, spans)
        mine["mrr@10"].append(rm.reciprocal_rank(relevant, 10))
        mine["ndcg@10"].append(rm.ndcg(relevant, len(spans), 10))
        mine["recall@5"].append(rm.evidence_recall(ranked, spans, 5))

    theirs = ranx.evaluate(ranx.Qrels(qrels), ranx.Run(run), list(mine))
    for metric, values in mine.items():
        assert sum(values) / len(values) == pytest.approx(theirs[metric], abs=1e-9), metric
