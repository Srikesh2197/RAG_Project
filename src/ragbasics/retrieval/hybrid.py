"""Hybrid retrieval: run dense and sparse search, then fuse the two ranked lists."""

import time
from typing import Any

from ragbasics.registry import register
from ragbasics.retrieval.base import Index, Retrieval
from ragbasics.retrieval.dense import DenseRetriever
from ragbasics.retrieval.fusion import NORMALISERS, reciprocal_rank_fusion, weighted_fusion
from ragbasics.retrieval.sparse import SparseRetriever
from ragbasics.types import Candidate


@register("retriever", "hybrid")
class HybridRetriever:
    """
    fusion: {method: rrf, k: 60}
            {method: weighted, weight: 0.7, normalise: minmax}
            `weight` is the dense list's share; the sparse list gets the rest.
    sparse: the sparse retriever's settings.
    depth:  how many chunks each retriever returns before fusion. Deeper than the number
            asked for, because a chunk at rank 120 in one list and rank 8 in the other
            can belong in the fused top 50, and a list cut at 50 would not show it.
    """

    def __init__(
        self,
        fusion: dict[str, Any] | None = None,
        sparse: dict[str, Any] | None = None,
        depth: int = 200,
    ):
        self.dense = DenseRetriever()
        self.sparse = SparseRetriever(**(sparse or {}))
        self.depth = depth
        settings = dict(fusion or {"method": "rrf"})
        self.method = settings.pop("method", "rrf")
        if self.method == "rrf":
            self.k = settings.pop("k", 60)
        elif self.method == "weighted":
            self.weight = settings.pop("weight")
            self.normalise = settings.pop("normalise", "minmax")
            if not 0 <= self.weight <= 1:
                raise ValueError(f"weight must be between 0 and 1; got {self.weight}")
            if self.normalise not in NORMALISERS:
                raise ValueError(f"unknown normaliser '{self.normalise}'")
        else:
            raise ValueError(f"unknown fusion method '{self.method}'; choose rrf or weighted")
        if settings:
            raise ValueError(f"unknown fusion settings for {self.method}: {sorted(settings)}")

    def fuse(self, lists: dict[str, list[Candidate]]) -> list[Candidate]:
        if self.method == "rrf":
            return reciprocal_rank_fusion(lists, self.k)
        weights = {"dense": self.weight, "sparse": 1 - self.weight}
        return weighted_fusion(lists, weights, self.normalise)

    def retrieve(self, question: str, k: int, index: Index) -> Retrieval:
        depth = max(self.depth, k)
        dense = self.dense.retrieve(question, depth, index)
        sparse = self.sparse.retrieve(question, depth, index)
        t0 = time.perf_counter()
        lists = {"dense": dense.candidates, "sparse": sparse.candidates}
        fused = self.fuse(lists)[:k]
        timings = dense.timings | sparse.timings | {"fuse": time.perf_counter() - t0}
        return Retrieval(fused, lists, timings, dense.embed_tokens)
