"""Fixed-size chunking: cut every `size` tokens, wherever that falls.

This is the naive baseline. It ignores sentences and paragraphs, so a boundary can land
in the middle of a sentence. Stage 3 measures what that costs.
"""

import tiktoken

from ragbasics.registry import register
from ragbasics.types import Chunk, Document


@register("chunker", "fixed")
class FixedChunker:
    def __init__(self, size: int = 512, encoding: str = "cl100k_base"):
        if size < 1:
            raise ValueError("size must be at least 1 token")
        self.size = size
        self.encoding = tiktoken.get_encoding(encoding)

    def chunk(self, document: Document) -> list[Chunk]:
        """Split `document` into consecutive chunks of `size` tokens; the last may be shorter.

        Chunks are slices of the canonical text, so they cover it end to end with no
        gaps and no overlap. Whitespace-only chunks are dropped.
        """
        text = document.text
        tokens = self.encoding.encode(text, disallowed_special=())
        if not tokens:
            return []
        # offsets[i] is the character position where token i starts.
        _, offsets = self.encoding.decode_with_offsets(tokens)
        # A character such as an emoji can span two tokens, which then share an offset;
        # the set removes the duplicate boundary.
        starts = sorted({0} | {offsets[i] for i in range(0, len(tokens), self.size)})
        bounds = [*starts, len(text)]

        chunks: list[Chunk] = []
        for start, end in zip(bounds, bounds[1:], strict=False):
            if not text[start:end].strip():
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{document.doc_id}:{len(chunks):04d}",
                    doc_id=document.doc_id,
                    text=text[start:end],
                    start_char=start,
                    end_char=end,
                    metadata=dict(document.metadata),
                )
            )
        return chunks
