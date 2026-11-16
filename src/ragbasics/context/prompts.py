"""The grounded-answer prompt.

Four parts: answer only from the passages, what to say when they fall short, the
passages (numbered), and the question. The baseline pastes the retrieved chunks in rank
order with no headers. Stage 8 can put a title, source and date line on each passage
(see context/assemble.py); the system prompt then says what that line is.

The faithfulness judge reads the passages through `render_passages` too, so it sees
exactly what the generator saw, headers included.
"""

from collections.abc import Sequence

from ragbasics.types import Chunk, Passage

ABSTAIN_ANSWER = "Insufficient information."

SYSTEM_PROMPT = f"""\
You answer questions about a collection of news articles. The user message contains \
passages retrieved from that collection, followed by a question.

Base your answer only on the passages. They are the only evidence you have, so do not \
add facts from memory. If the passages do not contain enough information to answer, \
reply with exactly: {ABSTAIN_ANSWER}

Give the answer alone, as briefly as it can be stated: a name, a yes or no, or one \
short sentence."""

# Stage 8: the same prompt, with one sentence saying what the first line of a passage is.
HEADER_SYSTEM_PROMPT = SYSTEM_PROMPT.replace(
    "followed by a question.",
    "followed by a question. Each passage starts with a line giving the title, source "
    "and publication date of the article it comes from.",
)


# The no-retrieval reference row (Stage 2): the same abstention and format lines, with
# the passages replaced by the model's own memory. It measures how much of the dataset
# the generator can answer without any retrieval.
CLOSED_BOOK_SYSTEM_PROMPT = f"""\
You answer questions about news articles published in late 2023. No articles are \
provided, so answer from your own knowledge.

If you do not know the answer, reply with exactly: {ABSTAIN_ANSWER}

Give the answer alone, as briefly as it can be stated: a name, a yes or no, or one \
short sentence."""


def build_closed_book_prompt(question: str) -> str:
    return f"Question: {question}"


def as_passages(context: Sequence[Chunk | Passage]) -> list[Passage]:
    """Plain chunks become one passage each, in the order given, with no header."""
    return [
        item if isinstance(item, Passage) else Passage(chunks=(item,), rank=i)
        for i, item in enumerate(context, start=1)
    ]


def passage_body(passage: Passage) -> str:
    """A passage as it appears between its tags: the header line, then the text."""
    text = passage.text.strip()
    return f"{passage.header}\n{text}" if passage.header else text


def render_passages(context: Sequence[Chunk | Passage]) -> str:
    """The numbered <passage> blocks. Without headers this is byte for byte the Stage 1
    format, so every cached generator and judge call stays valid."""
    passages = "\n\n".join(
        f'<passage id="{i}">\n{passage_body(passage)}\n</passage>'
        for i, passage in enumerate(as_passages(context), start=1)
    )
    return f"<passages>\n{passages}\n</passages>"


def build_user_prompt(question: str, context: Sequence[Chunk | Passage]) -> str:
    return f"{render_passages(context)}\n\nQuestion: {question}"


def system_prompt_for(context: Sequence[Chunk | Passage]) -> str:
    """The header sentence is added only when some passage carries a header."""
    has_header = any(p.header for p in as_passages(context))
    return HEADER_SYSTEM_PROMPT if has_header else SYSTEM_PROMPT


def build_prompt(question: str, context: Sequence[Chunk | Passage] | None) -> tuple[str, str]:
    """(system, user) for a question. `context=None` is the closed-book prompt."""
    if context is None:
        return CLOSED_BOOK_SYSTEM_PROMPT, build_closed_book_prompt(question)
    return system_prompt_for(context), build_user_prompt(question, context)
