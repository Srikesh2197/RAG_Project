"""Library equivalent of the hand-written BM25 index: bm25s.

    pip install -e ".[frameworks]"

bm25s computes the BM25 score of every (term, chunk) pair once, at index time, and
stores them in a sparse matrix; a query then adds up the rows of its terms. That is why
it is fast, and why `k1` and `b` are fixed when the index is built. The hand-written
index stores counts and applies `k1` and `b` at query time, which is slower per query
and lets a sweep reuse one index.

Its default method, "lucene", is the formula in `ragbasics.index.bm25`. Both sides here
are given the same terms, so the comparison is of the scoring and not of two tokenisers:
bm25s has its own tokeniser and can stem through the PyStemmer package (the Snowball
stemmer, a later revision of the Porter rules written by hand in this repo).

What differs:

- A term repeated in the question counts once in the hand-written index and once per
  occurrence in bm25s. The questions here are long and 393 of 441 repeat a term.
  Counting once scored higher on the pool questions (65.7 against 63.8 on recall in
  2,000 tokens), so that is what the repo does, and `search` below removes repeats
  before calling bm25s.
- bm25s returns `k` results even when fewer chunks share a term with the question; the
  rest have score 0. The hand-written index returns only chunks that matched.
- Chunks with equal scores come back in chunk order from the hand-written index and in
  no fixed order from bm25s.
"""

import bm25s


def build(documents: list[list[str]], k1: float = 1.2, b: float = 0.75) -> bm25s.BM25:
    index = bm25s.BM25(method="lucene", k1=k1, b=b)
    index.index(documents, show_progress=False)
    return index


def search(index: bm25s.BM25, terms: list[str], k: int) -> list[tuple[int, float]]:
    """(chunk number, score) for the `k` best chunks, best first."""
    query = list(dict.fromkeys(terms))
    chunks, scores = index.retrieve([query], k=k, show_progress=False)
    return [(int(i), float(s)) for i, s in zip(chunks[0], scores[0], strict=True) if s > 0]
