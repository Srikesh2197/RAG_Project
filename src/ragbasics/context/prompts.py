"""The grounded-answer prompt.

Four parts: answer only from the passages, what to say when they fall short, the
passages (numbered), and the question. The baseline pastes the retrieved chunks in rank
order with no source or date headers; Stage 8 measures what headers and ordering change.
"""

from ragbasics.types import Chunk

ABSTAIN_ANSWER = "Insufficient information."

SYSTEM_PROMPT = f"""\
You answer questions about a collection of news articles. The user message contains \
passages retrieved from that collection, followed by a question.

Base your answer only on the passages. They are the only evidence you have, so do not \
add facts from memory. If the passages do not contain enough information to answer, \
reply with exactly: {ABSTAIN_ANSWER}

Give the answer alone, as briefly as it can be stated: a name, a yes or no, or one \
short sentence."""


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


def build_user_prompt(question: str, chunks: list[Chunk]) -> str:
    passages = "\n\n".join(
        f'<passage id="{i}">\n{chunk.text.strip()}\n</passage>'
        for i, chunk in enumerate(chunks, start=1)
    )
    return f"<passages>\n{passages}\n</passages>\n\nQuestion: {question}"
