"""Dense retrieval: embed the question and find the nearest chunk vectors."""

import time

from ragbasics.embedding.base import Kind
from ragbasics.registry import register
from ragbasics.retrieval.base import Index, Retrieval


@register("retriever", "dense")
class DenseRetriever:
    def retrieve(self, question: str, k: int, index: Index, kind: Kind = "query") -> Retrieval:
        t0 = time.perf_counter()
        embedded = index.embedder.embed([question], kind=kind)
        t1 = time.perf_counter()
        found = index.store.search(embedded.vectors[0], k)
        t2 = time.perf_counter()
        return Retrieval(
            candidates=found,
            lists={"dense": found},
            timings={"embed_query": t1 - t0, "search": t2 - t1},
            embed_tokens=embedded.tokens,
        )
