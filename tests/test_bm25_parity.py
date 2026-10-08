"""The hand-written BM25 index against bm25s. Skipped when the `frameworks` extra is not
installed."""

import sys
from pathlib import Path

import pytest

pytest.importorskip("bm25s")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "framework_equivalents"))

import bm25_library  # noqa: E402

from ragbasics.index.bm25 import BM25Index  # noqa: E402
from ragbasics.retrieval.tokenize import Tokenizer  # noqa: E402

SENTENCES = [
    "Shares in the bakery chain rose nine percent on Tuesday after a strong quarter.",
    "The harbour ferry was delayed by fog and left at noon, two hours late.",
    "Volcanic ash closed the airport for three days, stranding thousands of travellers.",
    "The city council voted to plant four hundred trees along the harbour road.",
    "Fog also delayed flights at the airport, though the ferry ran on time on Wednesday.",
    "The bakery chain plans to open nine shops near the harbour before the end of the year.",
    "Council members said the airport road would be closed for tree planting.",
    "Travellers stranded by the ash were offered ferry tickets and bakery vouchers.",
]
# 40 texts of one to three sentences, each ending in a term of its own.
TEXTS = [" ".join(SENTENCES[i % 8 : i % 8 + 1 + i % 3]) + f" item{i}" for i in range(40)]
QUERIES = [
    "Why was the harbour ferry delayed?",
    "What closed the airport, and for how many days?",
    "Which bakery chain shares rose after a strong quarter?",
    "trees planted along the harbour road by the council item7",
    "Was the ferry late, and was the ferry delayed by fog at the harbour?",  # repeats terms
]


@pytest.mark.parametrize("k1,b", [(1.2, 0.75), (0.6, 0.0), (2.0, 1.0)])
@pytest.mark.parametrize("stemmer", ["none", "porter"])
def test_bm25_ranking_and_scores_match_bm25s(k1, b, stemmer):
    tokenizer = Tokenizer(stopwords=True, stemmer=stemmer)
    documents = [tokenizer(text) for text in TEXTS]
    mine = BM25Index.build(documents)
    theirs = bm25_library.build(documents, k1=k1, b=b)
    for query in QUERIES:
        terms = tokenizer(query)
        # Every chunk that matched, so a group of equal scores is never cut in two.
        ours = mine.search(terms, len(TEXTS), k1, b)
        library = bm25_library.search(theirs, terms, len(TEXTS))
        assert [score for _, score in ours] == pytest.approx(
            [score for _, score in library], rel=1e-4
        )
        # Chunks with equal scores come back in no fixed order from bm25s, so the order
        # is compared with ties put in chunk order on both sides.
        def tie_free(hits):
            return [i for i, _ in sorted(hits, key=lambda hit: (-round(hit[1], 4), hit[0]))]

        assert tie_free(ours) == tie_free(library)
