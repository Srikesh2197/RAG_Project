"""The structure-aware chunkers (Stage 3). No API calls: the semantic chunker is given
hand-made sentence vectors."""

import numpy as np
import pytest
import tiktoken

from ragbasics.chunking import units
from ragbasics.chunking.parent_child import ParentChildChunker
from ragbasics.chunking.recursive import ParagraphChunker, RecursiveChunker, SentenceChunker
from ragbasics.chunking.semantic import SemanticChunker, breakpoints
from ragbasics.embedding.base import Embedded
from ragbasics.pipeline import collapse_duplicates
from ragbasics.registry import register
from ragbasics.types import Candidate, Document

ENCODING = tiktoken.get_encoding("cl100k_base")


def tokens(text: str) -> int:
    return len(ENCODING.encode(text))


def texts(doc: Document, spans: list[units.Span]) -> list[str]:
    return [doc.text[start:end] for start, end in spans]


ARTICLE = Document(
    "doc_0",
    "Shares in the U.S. bakery chain rose on Tuesday. Dr. Lee called it \"a surprise.\" "
    "Was it?\n\n"
    "The ferry was delayed by fog. It left at noon.\n\n"
    "Volcanic ash closed the airport for three days, and the city council, which had met "
    "twice that week, voted to plant four hundred trees along the harbour road.",
    {"source": "Test"},
)


# --- Splitters ---------------------------------------------------------------------------


def test_sentences_split_on_punctuation_and_keep_abbreviations_whole():
    spans = units.sentences(ARTICLE.text, 0, len(ARTICLE.text))
    assert [t.strip() for t in texts(ARTICLE, spans)][:5] == [
        "Shares in the U.S. bakery chain rose on Tuesday.",
        'Dr. Lee called it "a surprise."',
        "Was it?",
        "The ferry was delayed by fog.",
        "It left at noon.",
    ]


@pytest.mark.parametrize(
    "splitter", [units.paragraphs, units.lines, units.sentences, units.words]
)
def test_every_splitter_tiles_its_input(splitter):
    spans = splitter(ARTICLE.text, 5, len(ARTICLE.text) - 5)
    assert "".join(texts(ARTICLE, spans)) == ARTICLE.text[5:-5]


def test_token_spans_cut_every_n_tokens():
    text = "one two three four five six seven"
    spans = units.token_spans(text, 0, len(text), 3, ENCODING)
    assert [text[a:b] for a, b in spans] == ["one two three", " four five six", " seven"]


# --- Packing -----------------------------------------------------------------------------

PIECES = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50)]


def test_pack_fills_each_chunk_up_to_the_size():
    # Sizes 4, 4, 4, 4, 4 into chunks of at most 10 tokens: two pieces each.
    assert units.pack(PIECES, [4] * 5, 10) == [(0, 20), (20, 40), (40, 50)]


def test_pack_overlap_repeats_whole_pieces_from_the_tail():
    # With 4 tokens of overlap each chunk starts with the last piece of the one before.
    assert units.pack(PIECES, [4] * 5, 10, overlap=4) == [(0, 20), (10, 30), (20, 40), (30, 50)]
    # 3 tokens of overlap is less than one piece, so nothing is repeated.
    assert units.pack(PIECES, [4] * 5, 10, overlap=3) == units.pack(PIECES, [4] * 5, 10)


# --- Recursive, paragraph and sentence chunkers ------------------------------------------


def test_recursive_keeps_paragraphs_whole_when_they_fit():
    chunks = RecursiveChunker(size=30).chunk(ARTICLE)
    assert chunks[0].text.strip().endswith("Was it?")
    assert chunks[1].text.strip() == "The ferry was delayed by fog. It left at noon."
    assert chunks[2].text.startswith("Volcanic ash")


def test_recursive_splits_an_oversize_paragraph_by_sentence_then_word():
    chunks = RecursiveChunker(size=14).chunk(ARTICLE)
    # The first paragraph is over 14 tokens, so it is cut at sentence ends.
    assert chunks[0].text.strip() == "Shares in the U.S. bakery chain rose on Tuesday."
    # The last paragraph is one long sentence, so it is cut between words.
    assert chunks[3].text == "Volcanic ash closed the airport for "
    assert all(tokens(c.text) <= 14 for c in chunks)


def test_sentence_chunker_ignores_paragraph_breaks():
    recursive = RecursiveChunker(size=24).chunk(ARTICLE)
    sentence = SentenceChunker(size=24).chunk(ARTICLE)
    # "Was it?" and the ferry sentence sit in different paragraphs. The sentence chunker
    # packs them into one chunk; the recursive one keeps the paragraph break.
    assert any("Was it?" in c.text and "ferry" in c.text for c in sentence)
    assert not any("Was it?" in c.text and "ferry" in c.text for c in recursive)


@pytest.mark.parametrize("chunker", [RecursiveChunker, ParagraphChunker, SentenceChunker])
@pytest.mark.parametrize("size", [3, 12, 40, 512])
def test_chunks_are_slices_that_cover_the_text_and_respect_the_size(chunker, size):
    doc = Document("doc_1", ARTICLE.text + "\n\nNaïve café — 日本語 😀 fin. " * 6 + "x" * 200)
    chunks = chunker(size=size).chunk(doc)
    assert "".join(c.text for c in chunks) == doc.text
    for c in chunks:
        assert doc.text[c.start_char : c.end_char] == c.text
        assert tokens(c.text) <= size + 1  # see the note on size in recursive.py


def test_overlap_repeats_text_and_still_covers_the_article():
    plain = RecursiveChunker(size=20).chunk(ARTICLE)
    lapped = RecursiveChunker(size=20, overlap=8).chunk(ARTICLE)
    assert sum(len(c.text) for c in lapped) > sum(len(c.text) for c in plain)
    assert lapped[0].start_char == 0 and lapped[-1].end_char == len(ARTICLE.text)
    for a, b in zip(lapped, lapped[1:], strict=False):
        assert b.start_char <= a.end_char  # no gap


def test_bad_sizes_are_refused():
    with pytest.raises(ValueError):
        RecursiveChunker(size=10, overlap=10)
    with pytest.raises(ValueError):
        ParentChildChunker(parent_size=128, child_size=128)


# --- Parent-child ------------------------------------------------------------------------


def test_parent_child_embeds_the_child_and_returns_the_parent():
    chunks = ParentChildChunker(parent_size=40, child_size=12).chunk(ARTICLE)
    parents = {c.chunk_id: c for c in RecursiveChunker(size=40).chunk(ARTICLE)}
    assert len(chunks) > len(parents)
    for c in chunks:
        parent = parents[c.metadata["parent_id"]]
        assert (c.text, c.start_char, c.end_char) == (
            parent.text, parent.start_char, parent.end_char)
        child = ARTICLE.text[c.metadata["child_start"] : c.metadata["child_end"]]
        assert c.text_to_embed == child and child in parent.text
        assert tokens(child) <= 12
    # The children of each parent cover it end to end.
    for parent in parents.values():
        kids = [c.text_to_embed for c in chunks if c.metadata["parent_id"] == parent.chunk_id]
        assert "".join(kids) == parent.text


def test_collapse_keeps_the_best_child_of_each_parent():
    chunks = ParentChildChunker(parent_size=40, child_size=12).chunk(ARTICLE)
    first, last = chunks[0], chunks[-1]
    sibling = next(c for c in chunks[1:] if c.metadata["parent_id"] == first.metadata["parent_id"])
    ranked = [Candidate(first, 0.9, 1), Candidate(sibling, 0.8, 2), Candidate(last, 0.7, 3)]
    kept = collapse_duplicates(ranked)
    assert [(c.chunk.chunk_id, c.rank) for c in kept] == [(first.chunk_id, 1), (last.chunk_id, 2)]


# --- Semantic ----------------------------------------------------------------------------


def test_breakpoints_fall_at_the_largest_distances():
    # Sentences 0-2 point one way, 3-4 another: the only real gap is after sentence 2.
    vectors = np.array([[1, 0], [1, 0.1], [1, 0], [0, 1], [0.1, 1]], dtype=np.float32)
    assert breakpoints(vectors, percentile=70) == [2]
    assert breakpoints(vectors[:1], percentile=70) == []


@register("embedder", "topic")
class TopicEmbedder:
    """Stands in for the embedder: one direction for bakery sentences, one for ferry ones."""

    model = "topic"

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, batch: list[str]) -> Embedded:
        self.calls += 1
        rows = [[1.0, 0.0] if "bakery" in text else [0.0, 1.0] for text in batch]
        return Embedded(np.asarray(rows, dtype=np.float32), tokens=10 * len(batch))


TOPICS = Document(
    "doc_t",
    "The bakery opened. The bakery sold bread. The bakery closed late. "
    "The ferry left. The ferry was late. The ferry returned.",
)


def test_semantic_chunker_cuts_where_the_topic_changes(tmp_path):
    chunker = SemanticChunker({"name": "topic"}, percentile=80, window=0, cache_dir=str(tmp_path))
    assert len(chunker.texts_to_embed([TOPICS])) == 6
    chunks = chunker.chunk(TOPICS)
    assert [c.text.strip() for c in chunks] == [
        "The bakery opened. The bakery sold bread. The bakery closed late.",
        "The ferry left. The ferry was late. The ferry returned.",
    ]
    assert "".join(c.text for c in chunks) == TOPICS.text
    assert chunker.billed_tokens == 60

    # Sentence vectors are cached on disk: a second chunker embeds nothing.
    again = SemanticChunker({"name": "topic"}, percentile=80, window=0, cache_dir=str(tmp_path))
    assert again.texts_to_embed([TOPICS]) == []
    assert [c.text for c in again.chunk(TOPICS)] == [c.text for c in chunks]
    assert again.embedder.calls == 0 and again.billed_tokens == 0


def test_semantic_segments_above_the_cap_are_packed_down():
    chunker = SemanticChunker({"name": "topic"}, percentile=80, window=0, max_size=8)
    chunks = chunker.chunk(TOPICS)
    assert len(chunks) > 2 and all(tokens(c.text) <= 9 for c in chunks)
    assert "".join(c.text for c in chunks) == TOPICS.text
