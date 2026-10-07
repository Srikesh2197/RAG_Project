"""Recursive chunking, and the sentence and paragraph chunkers as two settings of it.

The rule: split on the coarsest separator first (blank lines). Pieces that fit the size
are packed together into chunks. A piece that is still too big is split again on the next
separator (line, then sentence, then word, then every N tokens). So a boundary falls on
the largest natural break that keeps the chunk within the size.

    recursive   paragraph -> line -> sentence -> word -> token
    paragraph   paragraph -> sentence -> word -> token
    sentence    sentence -> word -> token   (paragraph breaks are ignored)

Size is the sum of the pieces' token counts, which is what LangChain's splitter measures
too. A chunk tokenised as one string can differ from that sum by a token or two.
"""

import tiktoken

from ragbasics.chunking import units
from ragbasics.chunking.units import Span, Splitter
from ragbasics.registry import register
from ragbasics.types import Chunk, Document

LEVELS: dict[str, Splitter] = {
    "paragraph": units.paragraphs,
    "line": units.lines,
    "sentence": units.sentences,
    "word": units.words,
}


@register("chunker", "recursive")
class RecursiveChunker:
    levels: tuple[str, ...] = ("paragraph", "line", "sentence", "word")

    def __init__(
        self,
        size: int = 512,
        overlap: int = 0,
        levels: list[str] | None = None,
        encoding: str = "cl100k_base",
    ):
        if size < 1:
            raise ValueError("size must be at least 1 token")
        if not 0 <= overlap < size:
            raise ValueError("overlap must be at least 0 tokens and smaller than size")
        self.size = size
        self.overlap = overlap
        self.splitters = [LEVELS[name] for name in (levels or self.levels)]
        self.encoding = tiktoken.get_encoding(encoding)

    def count(self, text: str, start: int, end: int) -> int:
        return len(self.encoding.encode(text[start:end], disallowed_special=()))

    def split(self, text: str, start: int, end: int, level: int = 0) -> list[Span]:
        """Chunk spans for text[start:end], starting from separator number `level`."""
        if level == len(self.splitters):
            return units.token_spans(text, start, end, self.size, self.encoding)
        chunks: list[Span] = []
        fitting: list[Span] = []  # consecutive pieces that each fit, waiting to be packed
        sizes: list[int] = []
        for piece in self.splitters[level](text, start, end):
            n = self.count(text, *piece)
            if n <= self.size:
                fitting.append(piece)
                sizes.append(n)
                continue
            # Too big for one chunk: close what came before it, then split it finer.
            chunks += units.pack(fitting, sizes, self.size, self.overlap)
            fitting, sizes = [], []
            chunks += self.split(text, *piece, level + 1)
        return chunks + units.pack(fitting, sizes, self.size, self.overlap)

    def chunk(self, document: Document) -> list[Chunk]:
        return units.to_chunks(document, self.split(document.text, 0, len(document.text)))


@register("chunker", "paragraph")
class ParagraphChunker(RecursiveChunker):
    """Whole paragraphs packed up to the size; an oversize paragraph is split by sentence."""

    levels = ("paragraph", "sentence", "word")


@register("chunker", "sentence")
class SentenceChunker(RecursiveChunker):
    """Whole sentences packed up to the size, wherever the paragraph breaks are."""

    levels = ("sentence", "word")
