"""What every query transformation takes and returns, and the settings they share.

A query transformation runs before retrieval and turns the question into one or more
queries. It decides nothing about ranking: the pipeline searches with the queries,
combines their lists and hands the result to the reranker, as before.

Three stages read a query, and each can be given a different one:

    retrieve_with   what the retriever searches with
    rerank_with     what the reranker scores each candidate against
                    each is one of
                      original   the question as asked
                      queries    the transformation's output in its place
                      both       the question, then the transformation's output
    (the generator always answers the question as asked)

Several queries give several ranked lists, and they have to become one:

    combine         for the retriever's lists
                      {method: rrf, k: 60}     reciprocal rank fusion, as in Stage 5:
                                               rewards chunks several queries found
                      {method: round_robin}    every query's first choice, then every
                                               query's second: no agreement needed
    rerank_combine  for the reranker's orderings of the same candidates
                      round_robin              as above
                      max                      a chunk's best score over the queries;
                                               needs scores that mean the same thing for
                                               every query, which a cross-encoder's do

    depth           how many chunks each query returns before the lists are combined,
                    when there are several. As in hybrid retrieval, deeper than the
                    number asked for: a chunk at rank 30 for two sub-questions can
                    belong in a combined top 20, and lists cut at 20 would not show it.

These settings are in each row's config file, so a row says what it did.
"""

from dataclasses import dataclass, field
from typing import Any, Protocol

from ragbasics.types import Query

ROUTES = ("original", "queries", "both")
COMBINERS = ("rrf", "round_robin")
RERANK_COMBINERS = ("round_robin", "max")


@dataclass
class Transformed:
    # The transformation's output. When it fell back, the question itself.
    queries: list[Query]
    timings: dict[str, float] = field(default_factory=dict)  # seconds per step
    # Tokens billed at list price for this call; 0 for a rule with no model.
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0  # LLM calls made for this question, cached or not
    cached: bool = False  # every reply came from the disk cache
    # Anything worth showing in the trace: a refused call, repairs to the reply, and
    # `fallback` when the question was used unchanged.
    notes: dict[str, Any] = field(default_factory=dict)


class Transformer(Protocol):
    name: str
    model: str
    retrieve_with: str
    rerank_with: str

    def transform(self, question: str) -> Transformed: ...


def route(question: str, queries: list[Query], which: str) -> list[Query]:
    """The queries one stage reads. The question is never listed twice."""
    original = Query(question, "original")
    if which == "original":
        return [original]
    others = [q for q in queries if q.text != question]
    if which == "both":
        return [original, *others]
    return queries or [original]


class Routing:
    """The settings every transformation shares. See the module docstring."""

    def __init__(
        self,
        retrieve_with: str = "queries",
        rerank_with: str = "original",
        combine: dict[str, Any] | None = None,
        rerank_combine: str = "round_robin",
        depth: int = 100,
    ):
        for label, value in (("retrieve_with", retrieve_with), ("rerank_with", rerank_with)):
            if value not in ROUTES:
                raise ValueError(f"unknown {label} '{value}'; choose from {ROUTES}")
        settings = dict(combine or {"method": "rrf"})
        self.combine_method = settings.pop("method", "rrf")
        if self.combine_method not in COMBINERS:
            raise ValueError(
                f"unknown combine method '{self.combine_method}'; choose from {COMBINERS}"
            )
        self.combine_k = settings.pop("k", 60) if self.combine_method == "rrf" else None
        if settings:
            raise ValueError(f"unknown combine settings for {self.combine_method}: "
                             f"{sorted(settings)}")
        if rerank_combine not in RERANK_COMBINERS:
            raise ValueError(
                f"unknown rerank_combine '{rerank_combine}'; choose from {RERANK_COMBINERS}"
            )
        self.retrieve_with = retrieve_with
        self.rerank_with = rerank_with
        self.rerank_combine = rerank_combine
        if depth < 1:
            raise ValueError(f"depth must be at least 1; got {depth}")
        self.depth = depth
