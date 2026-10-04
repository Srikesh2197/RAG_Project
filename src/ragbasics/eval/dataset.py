"""Turn the raw MultiHop-RAG files into documents, questions with gold spans, and frozen splits.

The raw dataset labels evidence as a sentence plus the article it came from. We convert
each one to a character span in the article, so "is this chunk relevant?" can be answered
for any chunking strategy by checking whether the chunk overlaps a gold span.
"""

import hashlib
import json
import random
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from ragbasics.config import DataConfig
from ragbasics.types import Document, EvidenceSpan, Question

METADATA_FIELDS = ("title", "source", "author", "category", "published_at", "url")


# --- Locating an evidence sentence inside an article -------------------------------------


def _collapse_whitespace(text: str) -> tuple[str, list[int]]:
    """Collapse each whitespace run to one space and strip the ends.

    Returns the collapsed string and, for each of its characters, the index of the
    character it came from in `text`.
    """
    chars: list[str] = []
    offsets: list[int] = []
    previous_was_space = True  # drops leading whitespace
    for i, ch in enumerate(text):
        if ch.isspace():
            if not previous_was_space:
                chars.append(" ")
                offsets.append(i)
            previous_was_space = True
        else:
            chars.append(ch)
            offsets.append(i)
            previous_was_space = False
    if chars and chars[-1] == " ":
        chars.pop()
        offsets.pop()
    return "".join(chars), offsets


def find_span(text: str, fact: str) -> tuple[int, int, str] | None:
    """Locate `fact` in `text`.

    Returns (start, end, method) with `end` exclusive, or None if it is not there.
    `method` is "exact" for a verbatim match and "whitespace" when the match only
    holds after collapsing whitespace. The first occurrence wins.
    """
    if not fact.strip():
        return None
    start = text.find(fact)
    if start != -1:
        return start, start + len(fact), "exact"

    collapsed_text, offsets = _collapse_whitespace(text)
    collapsed_fact, _ = _collapse_whitespace(fact)
    index = collapsed_text.find(collapsed_fact)
    if index == -1:
        return None
    return offsets[index], offsets[index + len(collapsed_fact) - 1] + 1, "whitespace"


# --- Raw rows to typed records ----------------------------------------------------------


def build_documents(raw_corpus: list[dict[str, Any]]) -> list[Document]:
    """One Document per article. Ids follow file order, which the pinned revision fixes."""
    return [
        Document(
            doc_id=f"doc_{i:04d}",
            text=row["body"],
            metadata={key: row.get(key) for key in METADATA_FIELDS},
        )
        for i, row in enumerate(raw_corpus)
    ]


@dataclass
class MappingReport:
    """What happened when evidence sentences were mapped to character spans."""

    questions_total: int = 0
    questions_without_evidence: int = 0
    evidence_total: int = 0
    matched_exact: int = 0
    matched_after_whitespace: int = 0
    article_not_in_corpus: int = 0
    sentence_not_in_article: int = 0
    duplicate_spans_dropped: int = 0
    sentence_repeats_in_article: int = 0
    incomplete_question_ids: list[str] = field(default_factory=list)

    @property
    def mapped(self) -> int:
        return self.matched_exact + self.matched_after_whitespace

    @property
    def mapping_rate(self) -> float:
        return self.mapped / self.evidence_total if self.evidence_total else 1.0


def build_questions(
    raw_questions: list[dict[str, Any]], documents: list[Document]
) -> tuple[list[Question], MappingReport]:
    """Convert raw questions, mapping each evidence sentence to a span in its article.

    A question with any evidence that cannot be mapped is still returned, and its id is
    listed in `report.incomplete_question_ids` so it can be kept out of the splits.
    """
    by_url = {doc.metadata["url"]: doc for doc in documents}
    by_title = {doc.metadata["title"]: doc for doc in documents}
    report = MappingReport(questions_total=len(raw_questions))
    questions: list[Question] = []

    for i, row in enumerate(raw_questions):
        question_id = f"q_{i:04d}"
        spans: list[EvidenceSpan] = []
        complete = True
        evidence_list = row.get("evidence_list") or []
        if not evidence_list:
            report.questions_without_evidence += 1

        for item in evidence_list:
            report.evidence_total += 1
            doc = by_url.get(item.get("url")) or by_title.get(item.get("title"))
            if doc is None:
                report.article_not_in_corpus += 1
                complete = False
                continue
            match = find_span(doc.text, item["fact"])
            if match is None:
                report.sentence_not_in_article += 1
                complete = False
                continue
            start, end, method = match
            if method == "exact":
                report.matched_exact += 1
                if doc.text.count(item["fact"]) > 1:
                    report.sentence_repeats_in_article += 1
            else:
                report.matched_after_whitespace += 1
            span = EvidenceSpan(doc.doc_id, start, end)
            if span in spans:
                report.duplicate_spans_dropped += 1
            else:
                spans.append(span)

        if not complete:
            report.incomplete_question_ids.append(question_id)
        questions.append(
            Question(
                question_id=question_id,
                text=row["query"],
                answer=row["answer"],
                question_type=row["question_type"],
                evidence=tuple(spans),
            )
        )
    return questions, report


# --- Evidence clusters ------------------------------------------------------------------


def evidence_clusters(questions: list[Question]) -> dict[str, str]:
    """Group questions that share a gold span, directly or through other questions.

    Returns question_id -> cluster id, where the id is the smallest question id in the
    cluster. A question with no evidence is a cluster of its own.

    This is connected components on a graph whose nodes are questions and gold spans,
    found with union-find.
    """
    parent: dict[object, object] = {}

    def find(node: object) -> object:
        parent.setdefault(node, node)
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:  # path compression
            parent[node], node = root, parent[node]
        return root

    for question in questions:
        for span in question.evidence:
            parent[find(span)] = find(question.question_id)

    members: dict[object, list[str]] = defaultdict(list)
    for question in questions:
        members[find(question.question_id)].append(question.question_id)
    return {
        question_id: min(group) for group in members.values() for question_id in group
    }


# --- Splits -----------------------------------------------------------------------------


def stratified_sample(
    questions: list[Question], n: int, seed: int
) -> tuple[list[Question], list[Question]]:
    """Draw `n` questions, keeping each question type's share of the whole.

    Returns (sample, rest), both in question-id order. Fractional quotas are settled by
    the largest-remainder method, so the sample has exactly `n` questions.
    """
    if n > len(questions):
        raise ValueError(f"cannot sample {n} from {len(questions)} questions")

    by_type: dict[str, list[Question]] = defaultdict(list)
    for question in sorted(questions, key=lambda q: q.question_id):
        by_type[question.question_type].append(question)

    exact = {t: n * len(group) / len(questions) for t, group in by_type.items()}
    quota = {t: int(share) for t, share in exact.items()}
    leftover = n - sum(quota.values())
    for t in sorted(exact, key=lambda t: (quota[t] - exact[t], t))[:leftover]:
        quota[t] += 1

    rng = random.Random(seed)
    chosen: set[str] = set()
    for t in sorted(by_type):
        group = by_type[t][:]
        rng.shuffle(group)
        chosen.update(q.question_id for q in group[: quota[t]])

    ordered = sorted(questions, key=lambda q: q.question_id)
    sample = [q for q in ordered if q.question_id in chosen]
    rest = [q for q in ordered if q.question_id not in chosen]
    return sample, rest


def evidence_disjoint_split(
    questions: list[Question],
    dev_size: int,
    test_size: int,
    max_cluster_share: float,
    seed: int,
) -> dict[str, list[Question]]:
    """Split so that dev and test never share a gold span. Needs `cluster` set on each question.

    1. Whole evidence clusters are assigned to the dev side or the test side, largest
       first, each to the side that is emptier relative to its target.
    2. A cluster may supply at most `max_cluster_share` of a split's answerable questions;
       larger clusters are randomly thinned to that cap.
    3. Each split is then a type-stratified sample from its side.
    Unanswerable questions have no evidence, so they are sampled at random, keeping
    their share of the whole.

    Returns dev, test, pool (dev-side leftovers: free to use during development) and
    test_pool (test-side leftovers: share evidence with test, so leave them alone).
    """
    answerable = [q for q in questions if q.evidence]
    unanswerable = [q for q in questions if not q.evidence]
    targets = {"dev": dev_size, "test": test_size}
    null_share = len(unanswerable) / len(questions)
    null_quota = {side: round(size * null_share) for side, size in targets.items()}
    quota = {side: targets[side] - null_quota[side] for side in targets}

    clusters: dict[str, list[Question]] = defaultdict(list)
    for question in sorted(answerable, key=lambda q: q.question_id):
        clusters[question.cluster].append(question)

    side_clusters: dict[str, list[str]] = {"dev": [], "test": []}
    filled = {"dev": 0, "test": 0}
    for cluster_id in sorted(clusters, key=lambda c: (-len(clusters[c]), c)):
        side = min(targets, key=lambda s: (filled[s] / quota[s], s))
        side_clusters[side].append(cluster_id)
        filled[side] += len(clusters[cluster_id])

    rng = random.Random(seed)
    sampled: dict[str, list[Question]] = {}
    leftover: dict[str, list[Question]] = {}
    for side in ("dev", "test"):
        cap = int(max_cluster_share * quota[side])
        candidates: list[Question] = []
        for cluster_id in side_clusters[side]:
            group = clusters[cluster_id][:]
            rng.shuffle(group)
            candidates += group[:cap]
        if len(candidates) < quota[side]:
            raise ValueError(
                f"{side} side has {len(candidates)} questions after capping clusters at "
                f"{cap}; need {quota[side]}"
            )
        sampled[side], _ = stratified_sample(candidates, quota[side], seed)
        taken = {q.question_id for q in sampled[side]}
        leftover[side] = [
            q for c in side_clusters[side] for q in clusters[c] if q.question_id not in taken
        ]

    null_dev, null_rest = stratified_sample(unanswerable, null_quota["dev"], seed)
    null_test, null_rest = stratified_sample(null_rest, null_quota["test"], seed + 1)

    def in_order(items: list[Question]) -> list[Question]:
        return sorted(items, key=lambda q: q.question_id)

    return {
        "dev": in_order(sampled["dev"] + null_dev),
        "test": in_order(sampled["test"] + null_test),
        "pool": in_order(leftover["dev"] + null_rest),
        "test_pool": in_order(leftover["test"]),
    }


def shared_gold_spans(a: list[Question], b: list[Question]) -> set[EvidenceSpan]:
    """Gold spans that appear in both question sets. Empty for an evidence-disjoint split."""
    return {s for q in a for s in q.evidence} & {s for q in b for s in q.evidence}


def largest_cluster_share(questions: list[Question]) -> float:
    """Share of the answerable questions that belong to the most common evidence cluster."""
    sizes = defaultdict(int)
    for question in questions:
        if question.evidence:
            sizes[question.cluster] += 1
    return max(sizes.values()) / sum(sizes.values()) if sizes else 0.0


# --- Reading and writing ----------------------------------------------------------------


def write_jsonl(path: Path, records: Iterable[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def read_documents(path: Path) -> list[Document]:
    with open(path) as f:
        return [Document(**json.loads(line)) for line in f]


def read_questions(path: Path) -> list[Question]:
    questions = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            row["evidence"] = tuple(EvidenceSpan(**span) for span in row["evidence"])
            questions.append(Question(**row))
    return questions


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(cfg: DataConfig) -> dict[str, Any]:
    """Build the processed files from the raw download and return the manifest.

    Writes documents.jsonl, questions_dev.jsonl, questions_test.jsonl, questions_pool.jsonl
    (dev-side leftovers, a reserve for prompt examples and judge calibration),
    questions_test_pool.jsonl (test-side leftovers), questions_excluded.jsonl (evidence
    could not be fully mapped) and manifest.json.
    """
    corpus_path = cfg.raw_dir / cfg.corpus_file
    questions_path = cfg.raw_dir / cfg.questions_file
    documents = build_documents(json.loads(corpus_path.read_text()))
    questions, report = build_questions(json.loads(questions_path.read_text()), documents)

    incomplete = set(report.incomplete_question_ids)
    excluded = [q for q in questions if q.question_id in incomplete]
    eligible = [q for q in questions if q.question_id not in incomplete]
    clusters = evidence_clusters(eligible)
    eligible = [replace(q, cluster=clusters[q.question_id]) for q in eligible]

    splits = evidence_disjoint_split(
        eligible, cfg.dev_size, cfg.test_size, cfg.max_cluster_share, cfg.seed
    )
    subset, _ = stratified_sample(splits["dev"], cfg.answer_subset_size, cfg.seed)
    subset_ids = {q.question_id for q in subset}
    splits["dev"] = [replace(q, answer_subset=q.question_id in subset_ids) for q in splits["dev"]]
    dev, test, pool = splits["dev"], splits["test"], splits["pool"]

    out = cfg.processed_dir
    write_jsonl(out / "documents.jsonl", documents)
    for name, split in splits.items():
        write_jsonl(out / f"questions_{name}.jsonl", split)
    write_jsonl(out / "questions_excluded.jsonl", excluded)

    manifest = {
        "repo_id": cfg.repo_id,
        "revision": cfg.revision,
        "raw_sha256": {
            cfg.corpus_file: sha256_file(corpus_path),
            cfg.questions_file: sha256_file(questions_path),
        },
        "seed": cfg.seed,
        "documents": len(documents),
        "questions": {
            "dev": len(dev),
            "dev_answer_subset": len(subset_ids),
            "test": len(test),
            "pool": len(pool),
            "test_pool": len(splits["test_pool"]),
            "excluded": len(excluded),
        },
        "split_check": {
            "evidence_clusters": len(set(clusters.values())),
            "gold_spans_shared_by_dev_and_test": len(shared_gold_spans(dev, test)),
            "gold_spans_shared_by_pool_and_test": len(shared_gold_spans(pool, test)),
            "largest_cluster_share": {
                "dev": largest_cluster_share(dev),
                "test": largest_cluster_share(test),
            },
        },
        "mapping": {**asdict(report), "mapped": report.mapped, "rate": report.mapping_rate},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
