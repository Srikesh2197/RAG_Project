"""An inverted index and BM25 scoring, written by hand.

An inverted index maps each term to the chunks that contain it (its postings), with the
count in each. Scoring a question then touches only the postings of the question's
terms, not every chunk.

BM25 scores a chunk as a sum over the question's terms of

    idf(term) * tf / (tf + k1 * (1 - b + b * length / average length))

- idf = ln(1 + (N - df + 0.5) / (df + 0.5)), where df is how many of the N chunks hold
  the term. A rare term counts for more. This is the form Lucene uses; it is never
  negative.
- tf is the term's count in the chunk. The fraction grows with tf and flattens towards
  1: `k1` sets how fast. At k1 = 0 only presence counts. (The textbook formula has a
  further factor (k1 + 1). It is the same for every term and chunk, so it changes no
  ranking, and Lucene and bm25s leave it out.)
- `b` corrects for length. A long chunk has more chances to contain any word; at b = 1
  its counts are scaled down in full proportion, at b = 0 length is ignored.

The index stores counts and lengths, not scores, so `k1` and `b` are chosen at query
time and trying another value needs no rebuild.
"""

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


class BM25Index:
    def __init__(
        self,
        vocabulary: dict[str, int],
        starts: np.ndarray,
        postings: np.ndarray,
        counts: np.ndarray,
        lengths: np.ndarray,
    ):
        self.vocabulary = vocabulary  # term -> term number
        # The postings of term t are postings[starts[t]:starts[t + 1]], in chunk order,
        # with the term's count in each chunk at the same positions of `counts`.
        self.starts = starts
        self.postings = postings
        self.counts = counts
        self.lengths = lengths  # terms per chunk
        self.average_length = float(lengths.mean()) if len(lengths) else 0.0

    def __len__(self) -> int:
        return len(self.lengths)

    @classmethod
    def build(cls, documents: list[list[str]]) -> "BM25Index":
        """`documents[i]` is the list of terms of chunk i."""
        vocabulary: dict[str, int] = {}
        by_term: list[list[tuple[int, int]]] = []  # term number -> [(chunk, count)]
        for i, terms in enumerate(documents):
            for term, count in Counter(terms).items():
                number = vocabulary.setdefault(term, len(vocabulary))
                if number == len(by_term):
                    by_term.append([])
                by_term[number].append((i, count))
        starts = np.zeros(len(by_term) + 1, dtype=np.int64)
        starts[1:] = np.cumsum([len(entries) for entries in by_term])
        flat = [entry for entries in by_term for entry in entries]
        postings = np.array([chunk for chunk, _ in flat], dtype=np.int32)
        counts = np.array([count for _, count in flat], dtype=np.float32)
        lengths = np.array([len(terms) for terms in documents], dtype=np.float32)
        return cls(vocabulary, starts, postings, counts, lengths)

    def document_frequency(self, term: str) -> int:
        number = self.vocabulary.get(term)
        return 0 if number is None else int(self.starts[number + 1] - self.starts[number])

    def idf(self, term: str) -> float:
        df = self.document_frequency(term)
        return float(np.log(1 + (len(self) - df + 0.5) / (df + 0.5)))

    def scores(self, terms: list[str], k1: float = 1.2, b: float = 0.75) -> np.ndarray:
        """The BM25 score of every chunk.

        A term repeated in the question counts once. Lucene and bm25s count it once per
        occurrence; on the long questions of this corpus that scored lower (Stage 5).
        """
        scores = np.zeros(len(self), dtype=np.float32)
        if not len(self):
            return scores
        length_factor = k1 * (1 - b + b * self.lengths / self.average_length)
        for term in dict.fromkeys(terms):
            number = self.vocabulary.get(term)
            if number is None:
                continue  # a term in no chunk cannot tell chunks apart
            entries = slice(self.starts[number], self.starts[number + 1])
            chunks, tf = self.postings[entries], self.counts[entries]
            scores[chunks] += self.idf(term) * tf / (tf + length_factor[chunks])
        return scores

    def search(
        self, terms: list[str], k: int, k1: float = 1.2, b: float = 0.75
    ) -> list[tuple[int, float]]:
        """(chunk number, score) for the `k` best chunks, best first.

        Chunks that share no term with the question score 0 and are never returned.
        Equal scores keep chunk order, so results are deterministic.
        """
        scores = self.scores(terms, k1, b)
        order = np.argsort(-scores, kind="stable")[:k]
        return [(int(i), float(scores[i])) for i in order if scores[i] > 0]

    def save(self, directory: Path, meta: dict[str, Any]) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.savez(
            directory / "postings.npz",
            starts=self.starts,
            postings=self.postings,
            counts=self.counts,
            lengths=self.lengths,
        )
        # Terms in number order, so the list is the vocabulary.
        (directory / "vocabulary.json").write_text(
            json.dumps(list(self.vocabulary), ensure_ascii=False)
        )
        (directory / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    @classmethod
    def load(cls, directory: Path) -> tuple["BM25Index", dict[str, Any]]:
        arrays = np.load(directory / "postings.npz")
        terms = json.loads((directory / "vocabulary.json").read_text())
        index = cls(
            {term: number for number, term in enumerate(terms)},
            arrays["starts"],
            arrays["postings"],
            arrays["counts"],
            arrays["lengths"],
        )
        return index, json.loads((directory / "meta.json").read_text())
