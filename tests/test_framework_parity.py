"""Hand-written components against their library equivalents. Skipped when the
`frameworks` extra is not installed."""

import sys
from pathlib import Path

import pytest

pytest.importorskip("langchain_text_splitters")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "framework_equivalents"))

from recursive_splitter import HAND_WRITTEN_LEVELS, langchain_chunks  # noqa: E402

from ragbasics.chunking.recursive import RecursiveChunker  # noqa: E402
from ragbasics.types import Document  # noqa: E402

PARAGRAPHS = [
    "Shares in the bakery chain rose nine percent on Tuesday after a strong quarter.",
    "The harbour ferry was delayed by fog. It left at noon, two hours late, and returned "
    "after dark with every seat taken.",
    "Volcanic ash closed the airport for three days, and the city council, which had met "
    "twice that week, voted to plant four hundred trees along the harbour road before "
    "the end of the year, a plan first proposed a decade ago.",
    "Short one.",
]
TEXT = "\n\n".join(PARAGRAPHS[i % 4] + f" Item {i}." for i in range(40))


@pytest.mark.parametrize("size,overlap", [(16, 0), (40, 0), (128, 0), (40, 8), (128, 26)])
def test_recursive_chunker_matches_langchain(size, overlap):
    chunker = RecursiveChunker(size=size, overlap=overlap, levels=HAND_WRITTEN_LEVELS)
    mine = [c.text.strip() for c in chunker.chunk(Document("doc", TEXT))]
    # LangChain emits the whitespace between two oversize paragraphs as a chunk of its
    # own; the hand-written chunker attaches it to the paragraph before.
    theirs = [c.strip() for c in langchain_chunks(TEXT, size, overlap) if c.strip()]
    assert mine == theirs
