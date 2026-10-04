from collections import Counter
from dataclasses import replace

import pytest

from ragbasics.eval.dataset import (
    build_documents,
    build_questions,
    evidence_clusters,
    evidence_disjoint_split,
    find_span,
    largest_cluster_share,
    read_documents,
    read_questions,
    shared_gold_spans,
    stratified_sample,
    write_jsonl,
)
from ragbasics.types import EvidenceSpan, Question

ARTICLE = "Alpha opened a store.  Beta closed\ntwo offices. Alpha opened a store."


def test_find_span_exact_returns_first_occurrence():
    start, end, method = find_span(ARTICLE, "Alpha opened a store.")
    assert (start, method) == (0, "exact")
    assert ARTICLE[start:end] == "Alpha opened a store."


def test_find_span_matches_across_whitespace_differences():
    fact = "Beta closed two offices."
    assert fact not in ARTICLE
    start, end, method = find_span(ARTICLE, fact)
    assert method == "whitespace"
    assert ARTICLE[start:end] == "Beta closed\ntwo offices."


def test_find_span_whitespace_match_spanning_a_double_space():
    start, end, _ = find_span(ARTICLE, "store. Beta")
    assert ARTICLE[start:end] == "store.  Beta"


@pytest.mark.parametrize("fact", ["Gamma did nothing.", "", "   "])
def test_find_span_returns_none_when_absent_or_empty(fact):
    assert find_span(ARTICLE, fact) is None


def _raw_corpus():
    return [
        {"title": "A", "url": "http://a", "source": "S1", "body": "One fact here. Other text."},
        {"title": "B", "url": "http://b", "source": "S2", "body": "Second fact there."},
    ]


def test_build_questions_maps_evidence_to_spans():
    documents = build_documents(_raw_corpus())
    raw = [
        {
            "query": "q?",
            "answer": "yes",
            "question_type": "comparison_query",
            "evidence_list": [
                {"url": "http://a", "title": "A", "fact": "One fact here."},
                {"url": "http://b", "title": "B", "fact": "Second fact there."},
                {"url": "http://a", "title": "A", "fact": "One fact here."},
            ],
        }
    ]
    questions, report = build_questions(raw, documents)

    assert questions[0].evidence == (
        EvidenceSpan("doc_0000", 0, 14),
        EvidenceSpan("doc_0001", 0, 18),
    )
    assert report.evidence_total == 3
    assert report.mapped == 3
    assert report.duplicate_spans_dropped == 1
    assert report.incomplete_question_ids == []


def test_build_questions_flags_unmappable_evidence():
    documents = build_documents(_raw_corpus())
    raw = [
        {
            "query": "q1?",
            "answer": "no",
            "question_type": "inference_query",
            "evidence_list": [
                {"url": "http://a", "title": "A", "fact": "Not in the article."},
                {"url": "http://missing", "title": "Missing", "fact": "One fact here."},
            ],
        },
        {
            "query": "q2?",
            "answer": "Insufficient information.",
            "question_type": "null_query",
            "evidence_list": [],
        },
    ]
    questions, report = build_questions(raw, documents)

    assert report.sentence_not_in_article == 1
    assert report.article_not_in_corpus == 1
    assert report.mapping_rate == 0.0
    assert report.incomplete_question_ids == ["q_0000"]
    assert report.questions_without_evidence == 1
    assert questions[1].evidence == ()


def test_build_questions_falls_back_to_title_when_url_differs():
    documents = build_documents(_raw_corpus())
    raw = [
        {
            "query": "q?",
            "answer": "yes",
            "question_type": "inference_query",
            "evidence_list": [{"url": "http://a?utm=1", "title": "A", "fact": "Other text."}],
        }
    ]
    questions, report = build_questions(raw, documents)
    assert questions[0].evidence == (EvidenceSpan("doc_0000", 15, 26),)
    assert report.mapped == 1


def test_evidence_clusters_join_questions_through_shared_spans():
    a, b, c, d = (EvidenceSpan("doc_0000", i, i + 5) for i in (0, 10, 20, 30))
    questions = [
        Question("q_0000", "?", "a", "t", (a, b)),
        Question("q_0001", "?", "a", "t", (b, c)),  # shares b with q_0000
        Question("q_0002", "?", "a", "t", (c,)),  # shares c with q_0001, nothing with q_0000
        Question("q_0003", "?", "a", "t", (d,)),
        Question("q_0004", "?", "a", "null_query"),
    ]
    assert evidence_clusters(questions) == {
        "q_0000": "q_0000",
        "q_0001": "q_0000",
        "q_0002": "q_0000",
        "q_0003": "q_0003",
        "q_0004": "q_0004",
    }


def _questions(counts: dict[str, int]) -> list[Question]:
    questions = []
    for question_type, count in counts.items():
        for _ in range(count):
            questions.append(Question(f"q_{len(questions):04d}", "q?", "a", question_type))
    return questions


def test_stratified_sample_keeps_type_proportions():
    questions = _questions({"a": 60, "b": 30, "c": 10})
    sample, rest = stratified_sample(questions, 20, seed=0)

    assert Counter(q.question_type for q in sample) == {"a": 12, "b": 6, "c": 2}
    assert len(rest) == 80
    assert {q.question_id for q in sample}.isdisjoint(q.question_id for q in rest)


def test_stratified_sample_hits_exact_size_with_fractional_quotas():
    questions = _questions({"a": 5, "b": 4, "c": 4})
    sample, rest = stratified_sample(questions, 7, seed=0)
    assert len(sample) == 7
    assert len(rest) == 6
    assert set(Counter(q.question_type for q in sample).values()) <= {2, 3}


def test_stratified_sample_is_deterministic_and_seed_dependent():
    questions = _questions({"a": 50, "b": 50})
    first, _ = stratified_sample(questions, 20, seed=1)
    again, _ = stratified_sample(list(reversed(questions)), 20, seed=1)
    other, _ = stratified_sample(questions, 20, seed=2)
    assert first == again
    assert first != other


def test_stratified_sample_rejects_oversized_request():
    with pytest.raises(ValueError):
        stratified_sample(_questions({"a": 3}), 4, seed=0)


def _clustered_questions() -> list[Question]:
    """30 questions sharing one hub span, 20 sharing another, 60 independent, 20 unanswerable."""
    types = ("comparison_query", "inference_query")
    questions: list[Question] = []

    def add(evidence: tuple[EvidenceSpan, ...], question_type: str | None = None) -> None:
        i = len(questions)
        questions.append(Question(f"q_{i:04d}", "?", "a", question_type or types[i % 2], evidence))

    hub_x, hub_y = EvidenceSpan("doc_x", 0, 10), EvidenceSpan("doc_y", 0, 10)
    for i in range(30):
        add((hub_x, EvidenceSpan("doc_x", 100 + 20 * i, 110 + 20 * i)))
    for i in range(20):
        add((hub_y, EvidenceSpan("doc_y", 100 + 20 * i, 110 + 20 * i)))
    for i in range(60):
        add((EvidenceSpan(f"doc_{i}", 0, 10),))
    for _ in range(20):
        add((), "null_query")

    clusters = evidence_clusters(questions)
    return [replace(q, cluster=clusters[q.question_id]) for q in questions]


def test_evidence_disjoint_split_shares_no_gold_span_between_dev_and_test():
    questions = _clustered_questions()
    splits = evidence_disjoint_split(
        questions, dev_size=30, test_size=30, max_cluster_share=0.2, seed=0
    )

    assert len(splits["dev"]) == 30
    assert len(splits["test"]) == 30
    assert shared_gold_spans(splits["dev"], splits["test"]) == set()
    assert shared_gold_spans(splits["pool"], splits["test"]) == set()

    ids = [q.question_id for split in splits.values() for q in split]
    assert sorted(ids) == [q.question_id for q in questions]


def test_evidence_disjoint_split_caps_clusters_and_keeps_unanswerable_share():
    splits = evidence_disjoint_split(
        _clustered_questions(), dev_size=30, test_size=30, max_cluster_share=0.2, seed=0
    )
    for name in ("dev", "test"):
        split = splits[name]
        # 20 of 130 questions are unanswerable: round(30 * 20 / 130) = 5 per split.
        assert sum(not q.evidence for q in split) == 5
        assert largest_cluster_share(split) <= 0.2


def test_evidence_disjoint_split_is_deterministic():
    questions = _clustered_questions()
    kwargs = dict(dev_size=30, test_size=30, max_cluster_share=0.2, seed=3)
    assert evidence_disjoint_split(questions, **kwargs) == evidence_disjoint_split(
        list(reversed(questions)), **kwargs
    )


def test_evidence_disjoint_split_fails_when_one_cluster_is_everything():
    hub = EvidenceSpan("doc_x", 0, 10)
    questions = [Question(f"q_{i:04d}", "?", "a", "t", (hub,), cluster="q_0000") for i in range(40)]
    with pytest.raises(ValueError, match="after capping"):
        evidence_disjoint_split(questions, dev_size=10, test_size=10, max_cluster_share=0.2, seed=0)


def test_jsonl_round_trip(tmp_path):
    documents = build_documents(_raw_corpus())
    questions = [
        Question(
            "q_0000",
            "q?",
            "a",
            "t",
            (EvidenceSpan("doc_0000", 0, 14),),
            cluster="q_0000",
            answer_subset=True,
        )
    ]
    write_jsonl(tmp_path / "docs.jsonl", documents)
    write_jsonl(tmp_path / "questions.jsonl", questions)

    assert read_documents(tmp_path / "docs.jsonl") == documents
    assert read_questions(tmp_path / "questions.jsonl") == questions
