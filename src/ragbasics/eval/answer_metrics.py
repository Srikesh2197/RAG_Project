"""Answer scoring that needs no LLM.

Reference answers here are short: "Yes", "No", a name, or "Insufficient information.".
A string comparison settles most answers for free and with certainty. Whatever it
cannot settle (a three-sentence answer, a name written differently) goes to the judge.
"""

import re
import string
from collections import Counter

from ragbasics.context.prompts import ABSTAIN_ANSWER
from ragbasics.types import Question

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCTUATION = str.maketrans("", "", string.punctuation)


def normalise(text: str) -> str:
    """Lower-case, drop punctuation and the articles a/an/the, collapse whitespace."""
    text = text.lower().translate(_PUNCTUATION)
    return " ".join(_ARTICLES.sub(" ", text).split())


_ABSTAIN = normalise(ABSTAIN_ANSWER)
_YES_NO = {"yes", "no"}


def is_abstention(answer: str) -> bool:
    """True if the answer opens or closes with the abstention phrase.

    "The passages cover only the first half. Insufficient information." declines.
    "Spotify, although the information is partial." does not.
    """
    text = normalise(answer)
    return text.startswith(_ABSTAIN) or text.endswith(_ABSTAIN)


def match_verdict(answer: str, reference: str) -> bool | None:
    """True or False when a string comparison settles it; None when a judge is needed."""
    a, r = normalise(answer), normalise(reference)
    if a == r:
        return True
    if is_abstention(answer):
        # Declining is right exactly when the reference declines.
        return is_abstention(reference)
    if a in _YES_NO and (r in _YES_NO or r == _ABSTAIN):
        # A bare yes or no against a different yes, no or abstention.
        return False
    # A longer answer, a name in another form, or a refusal in other words.
    return None


def fit_guesser(questions: list[Question]) -> dict[str, str]:
    """The most common reference answer for each question type.

    A guesser that reads only the question's type and gives this answer sets the floor
    for answer correctness: any system has to beat it to show it used the question.
    """
    by_type: dict[str, Counter[str]] = {}
    for question in questions:
        by_type.setdefault(question.question_type, Counter())[question.answer.strip()] += 1
    return {t: answers.most_common(1)[0][0] for t, answers in by_type.items()}
