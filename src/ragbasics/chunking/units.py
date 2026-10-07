"""Building blocks shared by the structure-aware chunkers.

A span is a pair (start, end) of character positions in a document's canonical text.
Every splitter here returns spans that tile their input: the first starts where the
input starts, each one begins where the previous ended, and the last ends where the
input ends. A separator (blank line, space) stays at the end of the piece before it.
Chunks built from these spans are therefore plain slices of the article, which is what
lets them be scored against gold evidence spans.
"""

import re
from collections.abc import Callable
from typing import Any

from ragbasics.types import Chunk, Document

Span = tuple[int, int]
Splitter = Callable[[str, int, int], list[Span]]

PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
LINE_BREAK = re.compile(r"\n")
WORD_BREAK = re.compile(r"\s+")
# Closing punctuation, any closing quotes or brackets, then whitespace; or a line break.
SENTENCE_BREAK = re.compile(r"""[.!?]+["'”’)\]]*\s+|\s*\n\s*""")
# A full stop after one of these does not end a sentence.
ABBREVIATIONS = frozenset(
    "mr mrs ms dr prof sr jr st vs inc ltd co corp no gen sen rep gov "
    "u.s u.k e.g i.e a.m p.m jan feb mar apr jun jul aug sep sept oct nov dec".split()
)


def _cut(text: str, start: int, end: int, cuts: list[int]) -> list[Span]:
    """Spans of text[start:end] between `cuts`. A whitespace-only span is joined to the
    one before it (or after it, at the very start), so no piece is ever empty of text."""
    bounds = sorted({start, end, *(c for c in cuts if start < c < end)})
    spans: list[Span] = []
    for a, b in zip(bounds, bounds[1:], strict=False):
        if spans and (not text[a:b].strip() or not text[spans[-1][0] : spans[-1][1]].strip()):
            spans[-1] = (spans[-1][0], b)
        else:
            spans.append((a, b))
    return spans


def _split_after(pattern: re.Pattern[str]) -> Splitter:
    def split(text: str, start: int, end: int) -> list[Span]:
        return _cut(text, start, end, [m.end() for m in pattern.finditer(text, start, end)])

    return split


paragraphs = _split_after(PARAGRAPH_BREAK)
lines = _split_after(LINE_BREAK)
words = _split_after(WORD_BREAK)


def _ends_sentence(text: str, start: int, match: re.Match[str]) -> bool:
    if "\n" in match.group():
        return True
    if match.group().lstrip()[:1] in "!?":
        return True
    # A full stop. Not a sentence end after an abbreviation or a single initial ("J."),
    # or when the next word starts in lower case.
    if match.end() < len(text) and text[match.end()].islower():
        return False
    before = text[max(start, match.start() - 12) : match.start()].split()
    word = before[-1].lower().lstrip("(\"'“‘") if before else ""
    return not (word in ABBREVIATIONS or (len(word) == 1 and word.isalpha()))


def sentences(text: str, start: int, end: int) -> list[Span]:
    """Sentence spans by rule: closing punctuation followed by whitespace, or a line break.

    It is a heuristic and makes mistakes ("U.S." at the end of a sentence joins it to
    the next one). A wrong boundary costs little here: it moves where a chunk may end.
    """
    cuts = [
        m.end() for m in SENTENCE_BREAK.finditer(text, start, end) if _ends_sentence(text, start, m)
    ]
    return _cut(text, start, end, cuts)


def token_spans(text: str, start: int, end: int, size: int, encoding: Any) -> list[Span]:
    """Cut text[start:end] every `size` tokens. The last resort for a piece with no
    usable separator, and the same rule the fixed chunker applies to a whole article."""
    tokens = encoding.encode(text[start:end], disallowed_special=())
    if not tokens:
        return []
    _, offsets = encoding.decode_with_offsets(tokens)
    cuts = [start + offsets[i] for i in range(size, len(tokens), size)]
    return _cut(text, start, end, cuts)


def pack(pieces: list[Span], sizes: list[int], size: int, overlap: int = 0) -> list[Span]:
    """Join neighbouring pieces into chunks of at most `size` tokens.

    `sizes[i]` is the token count of `pieces[i]`. Pieces are added until the next one
    would not fit. With `overlap`, the next chunk starts with the last whole pieces of
    this one, as many as fit in `overlap` tokens: the tail is repeated, never cut.
    """
    chunks: list[Span] = []
    run: list[int] = []  # indexes of the pieces in the chunk being built
    total = 0
    for i, n in enumerate(sizes):
        if run and total + n > size:
            chunks.append((pieces[run[0]][0], pieces[run[-1]][1]))
            # Drop pieces from the front until what is left is within the overlap and
            # leaves room for the next piece.
            while run and (total > overlap or total + n > size):
                total -= sizes[run.pop(0)]
        run.append(i)
        total += n
    if run:
        chunks.append((pieces[run[0]][0], pieces[run[-1]][1]))
    return chunks


def to_chunks(document: Document, spans: list[Span]) -> list[Chunk]:
    """Chunks for `spans` of `document`, numbered in order. Whitespace-only spans are dropped."""
    chunks: list[Chunk] = []
    for start, end in spans:
        if not document.text[start:end].strip():
            continue
        chunks.append(
            Chunk(
                chunk_id=f"{document.doc_id}:{len(chunks):04d}",
                doc_id=document.doc_id,
                text=document.text[start:end],
                start_char=start,
                end_char=end,
                metadata=dict(document.metadata),
            )
        )
    return chunks
