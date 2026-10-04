"""Data types shared by every stage."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Document:
    """One source document.

    `text` is the canonical text: every character offset in the project (gold evidence,
    and later chunk boundaries) refers to positions in this string. It is never edited
    after ingestion.
    """

    doc_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvidenceSpan:
    """Gold evidence: characters [start_char, end_char) of a document's canonical text."""

    doc_id: str
    start_char: int
    end_char: int


@dataclass(frozen=True)
class Question:
    question_id: str
    text: str
    answer: str
    question_type: str
    evidence: tuple[EvidenceSpan, ...] = ()
    # Questions that share gold evidence, directly or through a chain of other questions,
    # carry the same cluster id. Their results are correlated, so statistics resample
    # clusters, not single questions.
    cluster: str = ""
    # True for the development questions that also get answer-quality metrics.
    answer_subset: bool = False
