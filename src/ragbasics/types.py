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


@dataclass(frozen=True)
class Chunk:
    """A piece of a document: characters [start_char, end_char) of its canonical text.

    Keeping the offsets is what lets any chunk from any chunker be checked against the
    gold evidence spans.

    `text` is what is shown and sent to the generator. `embed_text` is what gets indexed;
    None means "the same as `text`". The two differ in later stages (parent-child and
    contextual retrieval).
    """

    chunk_id: str
    doc_id: str
    text: str
    start_char: int
    end_char: int
    embed_text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def text_to_embed(self) -> str:
        return self.text if self.embed_text is None else self.embed_text


@dataclass(frozen=True)
class Candidate:
    """A chunk returned by a retriever, with its score and its rank (1 is best)."""

    chunk: Chunk
    score: float
    rank: int
    retriever: str = "dense"


@dataclass
class Trace:
    """Everything that happened while answering one question.

    The app renders it and the evaluator (Stage 2) logs it. Later stages add fields for
    the steps they introduce.
    """

    question: str
    candidates: list[Candidate] = field(default_factory=list)
    # The chunks pasted into the prompt. Usually the top of `candidates`; the evaluator
    # retrieves deeper than it prompts, and can also supply gold evidence or nothing.
    context: list[Chunk] = field(default_factory=list)
    system_prompt: str = ""
    user_prompt: str = ""
    answer: str = ""
    stop_reason: str = ""
    models: dict[str, str] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)  # seconds per step
    usage: dict[str, int] = field(default_factory=dict)  # tokens per kind
    cost_usd: float = 0.0  # at list price, whether or not the answer came from the cache
    cached: bool = False
