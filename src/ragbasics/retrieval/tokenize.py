"""Turning text into the terms a keyword index matches on.

Keyword search matches strings, so every choice here decides what counts as "the same
word". Lowercasing makes "Apple" and "apple" one term. Dropping stopwords removes terms
that are in nearly every chunk. Stemming makes "acquired" and "acquiring" one term, and
also "university" and "universe". The chunks and the question must go through exactly
the same steps, or their terms do not meet.
"""

import re
from dataclasses import asdict, dataclass
from functools import cache

# A term is a run of letters and digits. "GPT-4" gives "gpt" and "4"; "U.S." gives "u"
# and "s". A possessive "'s" is dropped first, so "Amazon's" gives "amazon".
POSSESSIVE = re.compile(r"['’]s\b")
TERM = re.compile(r"[^\W_]+")

# Function words that carry no topic. BM25 already gives a term found in most chunks a
# weight near zero, so removing them changes scores little; it mainly shortens the work.
STOPWORDS = frozenset(
    "a about above after again against all am an and any are as at be because been "
    "before being below between both but by can did do does doing down during each few "
    "for from further had has have having he her here hers herself him himself his how "
    "i if in into is it its itself just me more most my myself no nor not now of off on "
    "once only or other our ours ourselves out over own same she should so some such "
    "than that the their theirs them themselves then there these they this those "
    "through to too under until up very was we were what when where which while who "
    "whom why will with you your yours yourself yourselves".split()
)

VOWELS = "aeiou"


def _consonant(word: str, i: int) -> bool:
    if word[i] in VOWELS:
        return False
    if word[i] == "y":
        return i == 0 or not _consonant(word, i - 1)
    return True


def _measure(stem: str) -> int:
    """Porter's m: how many vowel-then-consonant sequences the stem holds. "tr" and
    "tree" have 0, "trouble" has 1, "troubles" has 2. A suffix is removed only when the
    stem left behind is long enough by this count."""
    m, after_vowel = 0, False
    for i in range(len(stem)):
        consonant = _consonant(stem, i)
        if consonant and after_vowel:
            m += 1
        after_vowel = not consonant
    return m


def _has_vowel(stem: str) -> bool:
    return any(not _consonant(stem, i) for i in range(len(stem)))


def _double_consonant(word: str) -> bool:
    return len(word) >= 2 and word[-1] == word[-2] and _consonant(word, len(word) - 1)


def _cvc(word: str) -> bool:
    """Ends consonant, vowel, consonant, where the last is not w, x or y ("hop", "fil")."""
    n = len(word)
    return (
        n >= 3
        and _consonant(word, n - 3)
        and not _consonant(word, n - 2)
        and _consonant(word, n - 1)
        and word[-1] not in "wxy"
    )


STEP2 = [
    ("ational", "ate"), ("tional", "tion"), ("enci", "ence"), ("anci", "ance"),
    ("izer", "ize"), ("abli", "able"), ("alli", "al"), ("entli", "ent"), ("eli", "e"),
    ("ousli", "ous"), ("ization", "ize"), ("ation", "ate"), ("ator", "ate"),
    ("alism", "al"), ("iveness", "ive"), ("fulness", "ful"), ("ousness", "ous"),
    ("aliti", "al"), ("iviti", "ive"), ("biliti", "ble"),
]
STEP3 = [
    ("icate", "ic"), ("ative", ""), ("alize", "al"), ("iciti", "ic"), ("ical", "ic"),
    ("ful", ""), ("ness", ""),
]
STEP4 = [
    "al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement", "ment", "ent",
    "ion", "ou", "ism", "ate", "iti", "ous", "ive", "ize",
]


def _replace(word: str, rules: list[tuple[str, str]], min_measure: int) -> str:
    """Apply the first rule whose suffix matches, if the stem is long enough."""
    for suffix, replacement in rules:
        if word.endswith(suffix):
            stem = word[: -len(suffix)]
            return stem + replacement if _measure(stem) > min_measure else word
    return word


@cache
def porter_stem(word: str) -> str:
    """The Porter stemmer (1980): five rounds of suffix rules.

    It is not a dictionary. It does not know that "ran" is "run", and its output is
    often not a word ("happi", "univers"). That is fine: the stem only has to be the
    same string for the chunk and for the question.
    """
    if len(word) <= 2:
        return word

    # Step 1a: plurals.
    if word.endswith("sses") or word.endswith("ies"):
        word = word[:-2]
    elif word.endswith("s") and not word.endswith("ss"):
        word = word[:-1]

    # Step 1b: -eed, -ed, -ing.
    if word.endswith("eed"):
        if _measure(word[:-3]) > 0:
            word = word[:-1]
    else:
        for suffix in ("ed", "ing"):
            if word.endswith(suffix) and _has_vowel(word[: -len(suffix)]):
                word = word[: -len(suffix)]
                if word.endswith(("at", "bl", "iz")):
                    word += "e"
                elif _double_consonant(word) and word[-1] not in "lsz":
                    word = word[:-1]
                elif _measure(word) == 1 and _cvc(word):
                    word += "e"
                break

    # Step 1c: a final y after a vowel-bearing stem becomes i.
    if word.endswith("y") and _has_vowel(word[:-1]):
        word = word[:-1] + "i"

    word = _replace(word, STEP2, 0)
    word = _replace(word, STEP3, 0)

    # Step 4: remove a suffix outright when the stem is long (m > 1).
    for suffix in STEP4:
        if word.endswith(suffix):
            stem = word[: -len(suffix)]
            if _measure(stem) > 1 and (suffix != "ion" or stem.endswith(("s", "t"))):
                word = stem
            break

    # Step 5: a final e, and a doubled l.
    if word.endswith("e"):
        stem = word[:-1]
        m = _measure(stem)
        if m > 1 or (m == 1 and not _cvc(stem)):
            word = stem
    if _measure(word) > 1 and word.endswith("ll"):
        word = word[:-1]
    return word


STEMMERS = {"none": None, "porter": porter_stem}


@dataclass(frozen=True)
class Tokenizer:
    """Lowercasing is always on. Stopword removal and stemming are the two options."""

    stopwords: bool = False
    stemmer: str = "none"

    def __post_init__(self) -> None:
        if self.stemmer not in STEMMERS:
            raise ValueError(f"unknown stemmer '{self.stemmer}'; choose from {list(STEMMERS)}")

    def settings(self) -> dict[str, bool | str]:
        return asdict(self)

    def __call__(self, text: str) -> list[str]:
        terms = TERM.findall(POSSESSIVE.sub("", text.lower()))
        if self.stopwords:
            terms = [term for term in terms if term not in STOPWORDS]
        stem = STEMMERS[self.stemmer]
        return [stem(term) for term in terms] if stem else terms
