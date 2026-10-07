"""Library equivalent of the hand-written recursive chunker: LangChain's
RecursiveCharacterTextSplitter.

    pip install -e ".[frameworks]"

Set up to follow the same rule as `ragbasics.chunking.recursive.RecursiveChunker`:
length is counted in cl100k tokens, separators are blank line, line break and
whitespace, and each separator stays at the end of the piece before it.

What differs, and why the two can still disagree:

- LangChain has no sentence level. The comparison therefore runs the hand-written
  chunker with levels paragraph, line and word.
- LangChain returns strings. The hand-written chunker returns character spans, which is
  what the gold-evidence scoring needs. (`add_start_index` recovers a start by searching
  for the chunk's text, which picks the wrong place when a passage repeats.)
- LangChain emits the whitespace between two oversize paragraphs as a chunk of its own
  (its default `strip_whitespace=True` then drops it). The hand-written chunker attaches
  it to the paragraph before, so chunks tile the article. The parity test compares
  stripped text.
- LangChain splits a piece again when it is exactly the chunk size; the hand-written
  one keeps it. On the corpus the two give identical chunks for 609 of 609 articles at
  512 tokens, 608 at 256 and 575 at 128; every difference is this rule.
- A single word longer than the chunk size is kept whole by LangChain and cut every N
  tokens by the hand-written chunker.
"""

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

SEPARATORS = [r"\n\s*\n", r"\n", r"\s+"]
HAND_WRITTEN_LEVELS = ["paragraph", "line", "word"]


def langchain_chunks(text: str, size: int, overlap: int = 0) -> list[str]:
    encoding = tiktoken.get_encoding("cl100k_base")
    splitter = RecursiveCharacterTextSplitter(
        separators=SEPARATORS,
        is_separator_regex=True,
        keep_separator="end",
        strip_whitespace=False,
        chunk_size=size,
        chunk_overlap=overlap,
        length_function=lambda piece: len(encoding.encode(piece, disallowed_special=())),
    )
    return splitter.split_text(text)
