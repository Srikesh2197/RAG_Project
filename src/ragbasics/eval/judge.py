"""LLM judges for what a string comparison cannot settle.

Two questions, each with a yes-or-no verdict:

    correctness   does the answer say the same thing as the reference answer?
    faithfulness  does the answer follow from the retrieved passages alone?

They are independent. An answer recalled from the model's memory can be correct and
unfaithful; an answer that repeats a wrong passage is faithful and incorrect.

A judge is a model and makes mistakes. Its verdicts are checked two ways: against a
sample of human labels, and on every answer the string comparison already settled.
"""

import json
import re
from dataclasses import dataclass

from ragbasics.generation.llm import Generation, Generator
from ragbasics.types import Chunk

CORRECTNESS_SYSTEM = """\
You grade answers to questions about news articles. You are given a question, the \
reference answer, and a candidate answer. Decide whether the candidate gives the same \
answer as the reference.

Rules:
- Judge only the final answer the candidate commits to. Extra explanation is fine \
unless it contradicts that answer.
- A name counts in any common form (full name, surname, well-known abbreviation).
- If the reference is "Insufficient information.", the candidate is correct only if it \
declines to answer, in any wording. If the reference is anything else, a candidate that \
declines or does not commit to an answer is incorrect.
- Do not use your own knowledge of the facts. The reference is the truth here.

Reply with a JSON object and nothing else: \
{"reasoning": "<one sentence>", "verdict": "correct" or "incorrect"}"""

FAITHFULNESS_SYSTEM = """\
You check whether an answer is supported by a set of passages. You are given the \
passages, a question, and an answer. Decide whether a careful reader who had only these \
passages, and no other knowledge, could arrive at the answer.

Rules:
- "supported": the passages state or directly imply everything the answer asserts.
- "unsupported": any part of the answer needs a fact that is not in the passages, even \
if that fact is true.
- Do not judge whether the answer is correct in the real world.

Reply with a JSON object and nothing else: \
{"reasoning": "<one sentence>", "verdict": "supported" or "unsupported"}"""


@dataclass(frozen=True)
class Verdict:
    passed: bool | None  # None: the judge's reply could not be read
    reasoning: str
    generation: Generation


def parse_verdict(text: str, positive: str, negative: str) -> tuple[bool | None, str]:
    """Read {"reasoning": ..., "verdict": ...} from the reply, tolerating text around it."""
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    try:
        reply = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        reply = {}
    verdict = str(reply.get("verdict", "")).strip().lower()
    passed = True if verdict == positive else False if verdict == negative else None
    return passed, str(reply.get("reasoning", ""))


def correctness_prompt(question: str, reference: str, answer: str) -> str:
    return (
        f"Question: {question}\n\nReference answer: {reference}\n\nCandidate answer: {answer}"
    )


def faithfulness_prompt(question: str, answer: str, context: list[Chunk]) -> str:
    passages = "\n\n".join(
        f'<passage id="{i}">\n{chunk.text.strip()}\n</passage>'
        for i, chunk in enumerate(context, start=1)
    )
    return f"<passages>\n{passages}\n</passages>\n\nQuestion: {question}\n\nAnswer: {answer}"


def judge_correctness(llm: Generator, question: str, reference: str, answer: str) -> Verdict:
    generation = llm.generate(CORRECTNESS_SYSTEM, correctness_prompt(question, reference, answer))
    passed, reasoning = parse_verdict(generation.text, "correct", "incorrect")
    return Verdict(passed, reasoning, generation)


def judge_faithfulness(llm: Generator, question: str, answer: str, context: list[Chunk]) -> Verdict:
    generation = llm.generate(FAITHFULNESS_SYSTEM, faithfulness_prompt(question, answer, context))
    passed, reasoning = parse_verdict(generation.text, "supported", "unsupported")
    return Verdict(passed, reasoning, generation)
