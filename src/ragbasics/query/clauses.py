"""Decomposition with no model: cut the question into clauses by a fixed rule.

This is the control for the LLM decomposition. It is free, instant and never refused,
and it is crude: a clause is a slice of the question, so "while the other article says
the opposite" stays as it is, with nothing to say which article or the opposite of what.
A model can write sub-questions that stand on their own. Whether that is worth a paid
call per question is what the two are compared for.

The rule was written for the Stage 6 diagnostic (scripts/stage06_clauses.py), before any
Stage 7 result, and is used here unchanged.
"""

import re
import time
from typing import Any

from ragbasics.query.base import Routing, Transformed
from ragbasics.registry import register
from ragbasics.types import Query

MIN_WORDS = 5  # a piece shorter than this joins the piece before it
BOUNDARY = re.compile(r",\s+|\s+(?:while|whereas|and|but|before|after)\s+")


def split_clauses(question: str) -> list[str]:
    """Cut a question at commas and at a few joining words.

    A piece under five words (", 2023," or "and Fortune") is glued back onto the piece
    before it, so dates and short lists stay whole. A question with no boundary is one
    clause.
    """
    spans: list[list[int]] = []  # [start, end) of each clause in the question
    start = 0
    for match in [*BOUNDARY.finditer(question), None]:
        end = match.start() if match else len(question)
        if spans and len(question[start:end].split()) < MIN_WORDS:
            spans[-1][1] = end
        elif question[start:end].strip():
            spans.append([start, end])
        start = match.end() if match else end
    # A short first piece joins the one after it.
    if len(spans) > 1 and len(question[spans[0][0] : spans[0][1]].split()) < MIN_WORDS:
        spans[1][0] = spans[0][0]
        spans = spans[1:]
    return [question[a:b].strip() for a, b in spans]


@register("query_transform", "clauses")
class ClauseSplitter(Routing):
    name = "clauses"
    model = "rule"

    def __init__(self, max_queries: int | None = None, **routing: Any):
        super().__init__(**routing)
        self.max_queries = max_queries

    def transform(self, question: str) -> Transformed:
        started = time.perf_counter()
        clauses = split_clauses(question) or [question]
        notes: dict[str, Any] = {}
        if self.max_queries and len(clauses) > self.max_queries:
            notes["extra"] = len(clauses) - self.max_queries
            clauses = clauses[: self.max_queries]
        role = "clause" if len(clauses) > 1 else "original"
        return Transformed(
            queries=[Query(clause, role) for clause in clauses],
            timings={"transform": time.perf_counter() - started},
            cached=True,
            notes=notes,
        )
