"""Parent-child chunking: search small pieces, return the larger block around them.

A small chunk gives a sharper embedding; a large one gives the generator more to read.
This splits the two roles. Each article is cut into parents, and each parent into
children. One `Chunk` is emitted per child:

    embed_text   the child's text, which is what gets indexed and searched
    text         the parent's text, which is what is returned and sent to the generator
    start_char, end_char   the parent's span

So everything downstream (the prompt, the token budget, the gold-span check) sees the
parent, which is the text that was actually retrieved. Several children of one parent
can match the same question; the pipeline keeps the best-ranked one and drops the rest.
"""

from ragbasics.chunking.recursive import RecursiveChunker
from ragbasics.registry import register
from ragbasics.types import Chunk, Document


@register("chunker", "parent_child")
class ParentChildChunker:
    def __init__(self, parent_size: int = 512, child_size: int = 128):
        if child_size >= parent_size:
            raise ValueError("child_size must be smaller than parent_size")
        self.parents = RecursiveChunker(size=parent_size)
        self.children = RecursiveChunker(size=child_size)

    def chunk(self, document: Document) -> list[Chunk]:
        chunks: list[Chunk] = []
        for parent in self.parents.chunk(document):
            # Children are cut inside the parent, so each has exactly one parent.
            inside = Document(document.doc_id, parent.text)
            for n, child in enumerate(self.children.chunk(inside)):
                chunks.append(
                    Chunk(
                        chunk_id=f"{parent.chunk_id}:{n:02d}",
                        doc_id=document.doc_id,
                        text=parent.text,
                        start_char=parent.start_char,
                        end_char=parent.end_char,
                        embed_text=child.text,
                        metadata={
                            **document.metadata,
                            "parent_id": parent.chunk_id,
                            "child_start": parent.start_char + child.start_char,
                            "child_end": parent.start_char + child.end_char,
                        },
                    )
                )
        return chunks
