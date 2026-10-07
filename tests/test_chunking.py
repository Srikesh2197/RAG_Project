import pytest

from ragbasics.chunking.fixed import FixedChunker
from ragbasics.types import Document

# In cl100k_base this is 7 tokens: "one", " two", " three", " four", " five", " six", " seven".
SEVEN = Document("doc_0", "one two three four five six seven", {"source": "Test"})


def test_fixed_chunker_matches_hand_computed_example():
    chunks = FixedChunker(size=3).chunk(SEVEN)
    assert [c.text for c in chunks] == ["one two three", " four five six", " seven"]
    assert [(c.start_char, c.end_char) for c in chunks] == [(0, 13), (13, 27), (27, 33)]
    assert [c.chunk_id for c in chunks] == ["doc_0:0000", "doc_0:0001", "doc_0:0002"]


def test_chunk_offsets_point_into_the_canonical_text():
    for chunk in FixedChunker(size=2).chunk(SEVEN):
        assert SEVEN.text[chunk.start_char : chunk.end_char] == chunk.text


@pytest.mark.parametrize("size", [1, 2, 5, 512])
def test_chunks_cover_the_text_with_no_gap_and_no_overlap(size):
    # Accents, an emoji (one character, two tokens) and CJK exercise the offset mapping.
    doc = Document("doc_1", "Naïve café — 日本語 😀 fin. " * 20 + "end")
    chunks = FixedChunker(size=size).chunk(doc)
    assert "".join(c.text for c in chunks) == doc.text
    assert chunks[0].start_char == 0
    assert chunks[-1].end_char == len(doc.text)


def test_no_chunk_exceeds_the_size():
    chunker = FixedChunker(size=8)
    doc = Document("doc_2", "The quick brown fox jumps over the lazy dog. " * 30)
    for chunk in chunker.chunk(doc):
        assert len(chunker.encoding.encode(chunk.text)) <= 8


def test_short_and_empty_documents():
    assert len(FixedChunker(size=512).chunk(SEVEN)) == 1
    assert FixedChunker().chunk(Document("doc_3", "")) == []
    assert FixedChunker().chunk(Document("doc_4", "  \n ")) == []


def test_chunks_carry_document_metadata_and_embed_their_own_text():
    chunk = FixedChunker(size=3).chunk(SEVEN)[0]
    assert chunk.metadata == {"source": "Test"}
    assert chunk.text_to_embed == chunk.text


def test_overlap_repeats_the_tail_of_each_chunk():
    chunks = FixedChunker(size=3, overlap=1).chunk(SEVEN)
    assert [c.text for c in chunks] == ["one two three", " three four five", " five six seven"]
    for chunk in chunks:
        assert SEVEN.text[chunk.start_char : chunk.end_char] == chunk.text
    with pytest.raises(ValueError):
        FixedChunker(size=3, overlap=3)
