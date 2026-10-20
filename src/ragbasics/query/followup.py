"""Chat follow-ups: rewrite a message that leans on earlier turns into a question that
stands on its own.

"And what about Google?" contains nothing to search for. The retriever sees one string
and has no memory, so the missing words have to be put back before retrieval. One LLM
call reads the last few turns and the new message and writes the question in full.

Unlike the other transformations, the rewritten question also goes to the generator:
the message as typed cannot be answered from passages alone. The trace keeps both.

A message that already stands on its own should come back unchanged; rewriting it anyway
is the usual failure, since it can pull an earlier topic into a new question. The first
message of a chat has no history and is never sent to the model.

This corpus has no conversations with labels, so the rewriting is checked on a handful of
hand-written two-turn examples (scripts/stage07_followups.py) and is not in any results
table.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import ragbasics.generation.llm  # noqa: F401  (registers the generators)
from ragbasics.cache import CachedGenerator, DiskCache
from ragbasics.generation.llm import Generator
from ragbasics.query.llm import parse_lines
from ragbasics.registry import build, register

Turn = tuple[str, str]  # (question, answer)

SYSTEM_PROMPT = """\
You are given the last turns of a conversation about a collection of news articles, and \
a new message from the user. Rewrite the new message as a question that can be understood \
without the conversation.

Replace every word that points back at the conversation ("it", "they", "that company", \
"the same year", "what about") with what it refers to. Keep the user's wording otherwise, \
and do not answer the question or add facts that are not in the conversation. If the new \
message is already a complete question, or starts a new topic, reply with it unchanged.

Reply with the question alone, on one line."""


def build_prompt(history: list[Turn], message: str) -> str:
    turns = "\n\n".join(f"User: {question}\nAssistant: {answer}" for question, answer in history)
    return f"<conversation>\n{turns}\n</conversation>\n\nNew message: {message}"


@dataclass
class Standalone:
    question: str  # the message rewritten, or unchanged
    rewritten: bool = False
    timings: dict[str, float] = field(default_factory=dict)
    input_tokens: int = 0
    output_tokens: int = 0
    cached: bool = False
    notes: dict[str, Any] = field(default_factory=dict)


@register("followup", "llm")
class FollowUpRewriter:
    """
    generator  the LLM, as in a pipeline config.
    turns      how many earlier turns the model is shown.
    cache      the SQLite file of LLM calls.
    """

    def __init__(
        self,
        generator: dict[str, Any] | Generator,
        turns: int = 3,
        cache: str | Path | None = None,
    ):
        if turns < 1:
            raise ValueError(f"turns must be at least 1; got {turns}")
        if isinstance(generator, dict):
            generator = build("generator", generator["name"], **generator.get("params", {}))
        self.inner = generator
        self.model = generator.model
        self.turns = turns
        self._cache_path = Path(cache) if cache else None
        self._llm: Generator | None = None

    @property
    def llm(self) -> Generator:
        if self._llm is None:
            self._llm = (
                CachedGenerator(self.inner, DiskCache(self._cache_path))
                if self._cache_path
                else self.inner
            )
        return self._llm

    def rewrite(self, history: list[Turn], message: str) -> Standalone:
        if not history:
            return Standalone(message)
        started = time.perf_counter()
        try:
            generation = self.llm.generate(
                SYSTEM_PROMPT, build_prompt(history[-self.turns :], message)
            )
        except Exception as error:
            if type(error).__name__ != "BadRequestError":
                raise
            return Standalone(
                message, timings={"followup": time.perf_counter() - started},
                notes={"refused": True},
            )
        lines, _ = parse_lines(generation.text, limit=1)
        question = lines[0] if lines else message
        return Standalone(
            question,
            rewritten=question != message,
            timings={"followup": time.perf_counter() - started},
            input_tokens=generation.input_tokens,
            output_tokens=generation.output_tokens,
            cached=generation.cached,
            notes={} if lines else {"empty": True},
        )
