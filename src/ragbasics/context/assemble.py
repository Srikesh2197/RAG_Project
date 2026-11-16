"""Context assembly (Stage 8): turn a ranked list of chunks into the passages of a prompt.

    ranked candidates
      -> pool        the top `pool` candidates are the ones it may choose from
      -> dedupe      drop a chunk whose text repeats one already kept
      -> select      the top `top_k`, or `top_k` picked by maximal marginal relevance
      -> merge       neighbouring chunks of one article become one passage
      -> header      a title, source and date line on each passage
      -> budget      drop the lowest-ranked passages until the tokens fit
      -> order       rank order, best at the edges, or by article date

Every step is off by default, and then the result is the top `top_k` chunks in rank
order with no header: the prompt every stage before this one used.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from functools import cache
from typing import Any

import numpy as np

from ragbasics.context.mmr import min_max, mmr
from ragbasics.context.prompts import passage_body, render_passages
from ragbasics.registry import register
from ragbasics.retrieval.sparse import header as header_line
from ragbasics.types import Candidate, Chunk, Passage

ORDERS = ("rank", "edges", "chronological")

# Unit vectors for a list of chunks, one row each. The pipeline reads them from the store.
VectorLookup = Callable[[list[Chunk]], np.ndarray]


@cache
def _encoding() -> Any:
    import tiktoken

    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_encoding().encode(text, disallowed_special=()))


@dataclass
class Assembly:
    passages: list[Passage]
    notes: dict[str, Any] = field(default_factory=dict)

    @property
    def chunks(self) -> list[Chunk]:
        """Every chunk in the prompt, in the order the passages appear."""
        return [chunk for passage in self.passages for chunk in passage.chunks]


def normalised(text: str) -> str:
    return " ".join(text.lower().split())


def dedupe(candidates: list[Candidate]) -> tuple[list[Candidate], int]:
    """Keep the first chunk of each text, compared lowercased with whitespace collapsed.
    Repeated boilerplate ("Advertisement", a newsletter sign-up) and syndicated copies
    otherwise take a slot each. Returns the kept candidates and how many were dropped."""
    seen: set[str] = set()
    kept = []
    for candidate in candidates:
        key = normalised(candidate.chunk.text)
        if key not in seen:
            seen.add(key)
            kept.append(candidate)
    return kept, len(candidates) - len(kept)


def merge_neighbours(candidates: list[Candidate]) -> list[Passage]:
    """One passage per run of chunks from one article whose offsets touch or overlap.

    Chunks with a gap between them stay apart: joining them would hide that text is
    missing. A merged passage takes the best rank among its chunks. Returned in rank
    order."""
    by_doc: dict[str, list[Candidate]] = {}
    for candidate in candidates:
        by_doc.setdefault(candidate.chunk.doc_id, []).append(candidate)
    passages = []
    for group in by_doc.values():
        group.sort(key=lambda c: (c.chunk.start_char, c.chunk.end_char))
        run = [group[0]]
        end = group[0].chunk.end_char
        for candidate in group[1:]:
            if candidate.chunk.start_char <= end:
                run.append(candidate)
                end = max(end, candidate.chunk.end_char)
            else:
                passages.append(_passage(run))
                run, end = [candidate], candidate.chunk.end_char
        passages.append(_passage(run))
    return sorted(passages, key=lambda p: p.rank)


def _passage(run: list[Candidate]) -> Passage:
    return Passage(chunks=tuple(c.chunk for c in run), rank=min(c.rank for c in run))


def order_edges(passages: list[Passage]) -> list[Passage]:
    """The two best at the ends, the weakest in the middle: the layout suggested by "lost
    in the middle" (Liu et al., 2023), where models read the start and end of a long
    context best. The same rule as LangChain's LongContextReorder: walking from the
    weakest up, passages go alternately to the front and the back, so with an odd count
    the best is first and with an even count it is last, next to the question."""
    ranked = sorted(passages, key=lambda p: p.rank)
    out: list[Passage] = []
    for i, passage in enumerate(reversed(ranked)):
        if i % 2 == 1:
            out.append(passage)
        else:
            out.insert(0, passage)
    return out


def _published(passage: Passage) -> datetime | None:
    value = passage.metadata.get("published_at")
    try:
        return datetime.fromisoformat(str(value)) if value else None
    except ValueError:
        return None


def order_chronological(passages: list[Passage]) -> list[Passage]:
    """Articles oldest first, each article's passages in the order they appear in it.
    Articles with no date go last. A question about what happened before or after
    something then reads its evidence in time order."""

    def key(passage: Passage) -> tuple:
        date = _published(passage)
        return (date is None, date.timestamp() if date else 0.0, passage.doc_id,
                passage.chunks[0].start_char)

    return sorted(passages, key=key)


def prompt_header(metadata: dict[str, Any]) -> str:
    """The Stage 5 header line ("Title. Source, Month D, YYYY."), without the full stop it
    adds after a title that already ends in one ("...to what end?." becomes "...to what
    end?"). The retrieval indexes keep the Stage 5 form, so their cached scores hold."""
    line = header_line(metadata)
    title = str(metadata.get("title") or "")
    if title and title[-1] in "?!." and line.startswith(title + "."):
        line = title + line[len(title) + 1 :]
    return line


@register("context", "assembler")
class Assembler:
    """Builds the prompt's passages from a ranked candidate list.

    top_k         chunks in the prompt (the config's `top_k`)
    pool          candidates it may choose from; defaults to `top_k`. Only MMR and
                  dedupe can reach below `top_k`.
    dedupe        drop chunks whose text repeats one already kept
    mmr           None, or the lambda for maximal marginal relevance (1 = relevance only)
    merge         join neighbouring chunks of one article into one passage
    header        put "Title. Source, Month D, YYYY." above each passage
    token_budget  None, or a cap on the passages' cl100k tokens (headers included);
                  whole passages are dropped from the lowest rank up
    order         "rank", "edges" or "chronological"
    """

    def __init__(
        self,
        top_k: int,
        pool: int | None = None,
        dedupe: bool = False,
        mmr: float | None = None,
        merge: bool = False,
        header: bool = False,
        token_budget: int | None = None,
        order: str = "rank",
    ):
        if order not in ORDERS:
            raise ValueError(f"unknown order '{order}'; choose from {ORDERS}")
        if pool is not None and pool < top_k:
            raise ValueError(f"pool ({pool}) must be at least top_k ({top_k})")
        self.top_k = top_k
        self.pool = pool or top_k
        self.dedupe = dedupe
        self.mmr = mmr
        self.merge = merge
        self.header = header
        self.token_budget = token_budget
        self.order = order

    def settings(self) -> dict[str, Any]:
        return {
            "top_k": self.top_k, "pool": self.pool, "dedupe": self.dedupe, "mmr": self.mmr,
            "merge": self.merge, "header": self.header, "token_budget": self.token_budget,
            "order": self.order,
        }

    def labelled(self, passage: Passage) -> Passage:
        if not self.header:
            return passage
        return Passage(passage.chunks, passage.rank, prompt_header(passage.metadata))

    def wrap(self, chunks: list[Chunk]) -> list[Passage]:
        """Chunks given from outside the ranking (the gold-evidence reference row): one
        passage each, in the order given, with the header if it is on. Nothing is
        dropped, merged or reordered."""
        return [self.labelled(Passage((c,), i)) for i, c in enumerate(chunks, start=1)]

    def assemble(
        self, candidates: list[Candidate], vectors: VectorLookup | None = None
    ) -> Assembly:
        notes: dict[str, Any] = {}
        pool = list(candidates[: self.pool])
        if self.dedupe:
            pool, notes["duplicates"] = dedupe(pool)

        if self.mmr is not None and len(pool) > self.top_k:
            if vectors is None:
                raise ValueError("MMR needs the chunks' vectors; none were given.")
            matrix = vectors([c.chunk for c in pool])
            relevance = min_max(np.array([c.score for c in pool]))
            picked = mmr(relevance, matrix @ matrix.T, self.top_k, self.mmr)
            # How far below the top_k the diversity pick reached.
            notes["mmr_swapped"] = sum(i >= self.top_k for i in picked)
            selected = [pool[i] for i in picked]
        else:
            selected = pool[: self.top_k]

        if self.merge:
            passages = merge_neighbours(selected)
            notes["merged"] = len(selected) - len(passages)
        else:
            passages = [Passage((c.chunk,), c.rank) for c in sorted(selected, key=lambda c: c.rank)]
        passages = [self.labelled(p) for p in passages]

        if self.token_budget is not None:
            kept, used = [], 0
            for passage in passages:  # rank order
                tokens = count_tokens(passage_body(passage))
                if used + tokens > self.token_budget and kept:
                    break
                kept.append(passage)
                used += tokens
            notes["over_budget"] = len(passages) - len(kept)
            passages = kept

        if self.order == "edges":
            passages = order_edges(passages)
        elif self.order == "chronological":
            passages = order_chronological(passages)

        notes |= {
            "order": self.order,
            "chunks": sum(len(p.chunks) for p in passages),
            "passages": len(passages),
            "tokens": count_tokens(render_passages(passages)),
        }
        return Assembly(passages, notes)
