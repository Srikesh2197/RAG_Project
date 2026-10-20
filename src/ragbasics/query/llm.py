"""Query transformations written by an LLM: rewriting, multi-query, HyDE and
decomposition into sub-questions.

Each is one call with the question as the whole user message. The model sees no passage
and no earlier turn. What is written here, where it can be read:

- the four prompts;
- parsing: a reply is text, and a list of queries has to be read out of it. Numbering,
  bullets, quotation marks and a "Query:" label are stripped; blank lines and repeats
  are dropped; lines beyond the number asked for are cut;
- the fallback: when the provider refuses the prompt, or nothing usable is left after
  parsing, the question is used unchanged and the trace says so. Any other error is
  raised;
- the count of paid calls as they complete (`meter`), so a run that crashes can still
  write what it spent.

What each transformation is for is in its own docstring.
"""

import re
import time
from pathlib import Path
from typing import Any

import ragbasics.generation.llm  # noqa: F401  (registers the generators)
from ragbasics.cache import CachedGenerator, DiskCache
from ragbasics.costs import Meter
from ragbasics.generation.llm import Generator
from ragbasics.query.base import Routing, Transformed
from ragbasics.registry import build, register
from ragbasics.types import Query

BULLET = re.compile(r"^\s*(?:[-*•]+|\(?\d+[.):]|[Qq]\d+[.):])\s*")
LABEL = re.compile(
    r"^(?:(?:rewritten|search|sub)[- ]?)?(?:query|question|passage|phrasing)\s*\d*\s*:\s*",
    re.IGNORECASE,
)
QUOTES = "\"'“”‘’`"


def clean_line(line: str) -> str:
    """One line of a reply without its numbering, bullet, label or quotation marks."""
    line = BULLET.sub("", line.strip())
    line = LABEL.sub("", line).strip()
    if len(line) > 1 and line[0] in QUOTES and line[-1] in QUOTES:
        line = line[1:-1].strip()
    return line


def parse_lines(reply: str, limit: int) -> tuple[list[str], dict[str, int]]:
    """The queries in a reply that was asked for one per line, and what was repaired.

    Blank lines are skipped, a line that repeats an earlier one (ignoring case and
    spacing) counts once, and lines beyond `limit` are cut.
    """
    lines: list[str] = []
    seen: set[str] = set()
    repeated = 0
    for raw in reply.splitlines():
        line = clean_line(raw)
        if not line:
            continue
        key = " ".join(line.lower().split())
        if key in seen:
            repeated += 1
        else:
            seen.add(key)
            lines.append(line)
    extra = max(0, len(lines) - limit)
    return lines[:limit], {"repeated": repeated, "extra": extra}


def parse_passage(reply: str) -> str:
    """A reply that should be one passage, as one line of text."""
    return " ".join(clean_line(line) for line in reply.splitlines() if line.strip())


class LLMTransformer(Routing):
    """
    generator  the LLM, as in a pipeline config: {name: openai, params: {model: ...}}.
    cache      the SQLite file of LLM calls; a repeated question is not paid for again.
    The routing settings are described in query/base.py.
    """

    name = ""
    role = ""
    # Output tokens billed per call, for the estimate printed before a run. The model
    # reasons before it replies and bills that as output, so this is a measured figure
    # (the mean over scripts/stage07_transform.py --examples 5, gpt-6-luna at its
    # default effort) and not the length of the reply.
    output_tokens = 130

    def __init__(
        self,
        generator: dict[str, Any] | Generator,
        cache: str | Path | None = None,
        **routing: Any,
    ):
        super().__init__(**routing)
        if isinstance(generator, dict):
            generator = build("generator", generator["name"], **generator.get("params", {}))
        self.inner = generator
        self.model = generator.model
        self._cache_path = Path(cache) if cache else None
        self._llm: Generator | None = None
        self.meter = Meter()

    @property
    def llm(self) -> Generator:
        # The cache file is opened on first use, so building a pipeline creates none.
        if self._llm is None:
            self._llm = (
                CachedGenerator(self.inner, DiskCache(self._cache_path))
                if self._cache_path
                else self.inner
            )
        return self._llm

    def system_prompt(self) -> str:
        raise NotImplementedError

    def prompt(self, question: str) -> str:
        return f"Question: {question}"

    def parse(self, reply: str, question: str) -> tuple[list[str], dict[str, int]]:
        raise NotImplementedError

    def is_cached(self, question: str) -> bool:
        check = getattr(self.llm, "is_cached", None)
        return bool(check and check(self.system_prompt(), self.prompt(question)))

    def fallback(self, question: str, started: float, **notes: Any) -> Transformed:
        return Transformed(
            queries=[Query(question, "original")],
            timings={"transform": time.perf_counter() - started},
            notes={"fallback": True, **notes},
        )

    def transform(self, question: str) -> Transformed:
        started = time.perf_counter()
        try:
            generation = self.llm.generate(self.system_prompt(), self.prompt(question))
        except Exception as error:
            # The OpenAI and Anthropic SDKs both raise BadRequestError when the request
            # itself is rejected, as a usage-policy filter does. Named by its class so
            # neither SDK is imported here. Nothing is cached for a refused question.
            if type(error).__name__ != "BadRequestError":
                raise
            return self.fallback(question, started, refused=True)
        if not generation.cached:
            self.meter.add(self.model, generation.input_tokens, generation.output_tokens)
        texts, repairs = self.parse(generation.text, question)
        notes = {k: v for k, v in repairs.items() if v} | {"stop_reason": generation.stop_reason}
        if texts:
            result = Transformed(
                queries=[
                    Query(text, "original" if text == question else self.role) for text in texts
                ],
                timings={"transform": time.perf_counter() - started},
                notes=notes,
            )
        else:
            # An empty reply, or one cut off before any text: usually the output limit
            # was spent on reasoning.
            result = self.fallback(question, started, empty=True, **notes)
        result.input_tokens = generation.input_tokens
        result.output_tokens = generation.output_tokens
        result.calls, result.cached = 1, generation.cached
        return result


@register("query_transform", "rewrite")
class Rewriter(LLMTransformer):
    """One query in place of the question, worded for search.

    Aimed at questions that are badly worded for retrieval: chatty, misspelt, or wrapped
    in words no document contains. It cannot add information the question lacks, and it
    can lose some: a rewrite that drops a date or a source has made the query worse.
    """

    name = "rewrite"
    role = "rewrite"

    def system_prompt(self) -> str:
        return (
            "You rewrite a question as a search query for a collection of news articles.\n\n"
            "Keep every name, source, date, number and other detail the question gives. "
            "Drop words that are about the asking and would not appear in an article, such "
            'as "according to", "does the article suggest" or "which of the following". Do '
            "not answer the question, and do not add any fact it does not state.\n\n"
            "Reply with the rewritten query alone, on one line."
        )

    def parse(self, reply: str, question: str) -> tuple[list[str], dict[str, int]]:
        lines, repairs = parse_lines(reply, limit=1)
        return lines, repairs


@register("query_transform", "multi_query")
class MultiQuery(LLMTransformer):
    """Several phrasings of the question; each is searched and the lists are combined.

    Aimed at wording: a chunk one phrasing misses, another may reach. Every phrasing is
    after the same chunks, so the lists are fused by reciprocal rank, and the question
    itself is usually searched too (`retrieve_with: both`). The risk is drift: a
    paraphrase that asks something slightly different brings in its own chunks.
    """

    name = "multi_query"
    role = "paraphrase"

    def __init__(self, generator: dict[str, Any] | Generator, n: int = 3, **settings: Any):
        if n < 1:
            raise ValueError(f"n must be at least 1; got {n}")
        settings.setdefault("retrieve_with", "both")
        super().__init__(generator, **settings)
        self.n = n
        self.output_tokens = 80 * n  # measured: 232 for three phrasings

    def system_prompt(self) -> str:
        return (
            f"You write {self.n} other ways of asking a question, to search a collection of "
            "news articles with.\n\n"
            "Each must ask for the same information as the question and keep every name, "
            "source, date and number in it. Change the wording and the order of the "
            "sentence, so that each could match articles that use different words. Do not "
            "answer the question, and do not add any fact it does not state.\n\n"
            f"Reply with exactly {self.n} lines, one phrasing per line, with no numbering "
            "and no other text."
        )

    def parse(self, reply: str, question: str) -> tuple[list[str], dict[str, int]]:
        lines, repairs = parse_lines(reply, limit=self.n)
        kept = [line for line in lines if line != question]
        return kept, repairs | {"missing": self.n - len(kept)}


@register("query_transform", "hyde")
class HyDE(LLMTransformer):
    """A made-up passage that would answer the question; the passage is searched with.

    HyDE stands for hypothetical document embeddings. A question and the passage that
    answers it are different kinds of text; a made-up answer is the same kind of text
    as the real one, so its vector may sit nearer to it. The passage is embedded as a
    document, not as a query.

    It does not have to be true to work: a wrong name in the right sort of sentence can
    still land near the right article. It also can mislead, since the search then
    follows what the model made up. And where the model knows the real answer from its
    training data, the passage carries that answer into the search, which a corpus the
    model has never seen would not allow.
    """

    name = "hyde"
    role = "hypothetical"
    output_tokens = 490  # a reply of about 60 tokens after far more reasoning

    def system_prompt(self) -> str:
        return (
            "You write a short passage of the kind a news article would contain, which "
            "would answer a question.\n\n"
            "Write two to four sentences in the plain style of a news report. State the "
            "facts the question asks about as if you knew them, one sentence for each "
            "fact it needs. Use the names, sources, dates and numbers the question gives. "
            "The passage is used only to search for real articles, so it does not have to "
            'be correct, but it must be specific. Do not write "the article says" and do '
            "not mention that the passage is made up.\n\n"
            "Reply with the passage alone."
        )

    def parse(self, reply: str, question: str) -> tuple[list[str], dict[str, int]]:
        passage = parse_passage(reply)
        return ([passage] if passage else []), {}


@register("query_transform", "decompose")
class Decomposer(LLMTransformer):
    """One sub-question per fact the question needs.

    Aimed at questions that ask about several things at once. One vector for the whole
    question sits between its parts, and a reranker scoring a chunk against the whole
    question marks down a chunk that supports one part only (Stage 6). Each
    sub-question is searched for, or scored against, on its own. Sub-questions are
    after different chunks, so their lists are combined by round robin by default.
    """

    name = "decompose"
    role = "sub-question"

    def __init__(
        self, generator: dict[str, Any] | Generator, max_queries: int = 4, **settings: Any
    ):
        if max_queries < 2:
            raise ValueError(f"max_queries must be at least 2; got {max_queries}")
        settings.setdefault("combine", {"method": "round_robin"})
        super().__init__(generator, **settings)
        self.max_queries = max_queries
        self.output_tokens = 220

    def system_prompt(self) -> str:
        return (
            "You break a question about news articles into the separate facts that have "
            "to be looked up to answer it.\n\n"
            f"Write one sub-question for each fact, and at most {self.max_queries}. Each "
            "sub-question must make sense without the others: repeat the names, sources "
            'and dates it needs, and do not refer back with words such as "it", "they" or '
            '"the other article". Do not answer anything, and do not add any fact the '
            "question does not state. If the question asks for one fact only, reply with "
            "the question unchanged.\n\n"
            "Reply with one sub-question per line, with no numbering and no other text."
        )

    def parse(self, reply: str, question: str) -> tuple[list[str], dict[str, int]]:
        return parse_lines(reply, limit=self.max_queries)
