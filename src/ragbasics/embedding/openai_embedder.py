"""OpenAI embedding models.

They take no query or document prefix and return vectors of length 1. The
`text-embedding-3` models are trained so that the leading dimensions carry most of the
meaning. `dimensions` here cuts the full vector and re-normalises it, which is what the
API's own `dimensions` parameter does; cutting locally means one paid call serves every
length.
"""

from typing import Any

import numpy as np

from ragbasics.embedding.base import CachingEmbedder, Embedded, Kind
from ragbasics.registry import register


@register("embedder", "openai")
class OpenAIEmbedder(CachingEmbedder):
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        dimensions: int | None = None,
        batch_size: int = 100,
        client: Any = None,
    ):
        self.model = model
        self.dimensions = dimensions
        self.batch_size = batch_size
        self._client = client

    @property
    def client(self) -> Any:
        # Created on first use, so building a pipeline does not need an API key.
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI()
        return self._client

    def cache_identity(self, kind: Kind) -> str:
        return f"openai:{self.model}"  # the same vector for a query and a document

    def _embed(self, texts: list[str], kind: Kind) -> Embedded:
        rows: list[list[float]] = []
        tokens = 0
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            response = self.client.embeddings.create(model=self.model, input=batch)
            rows += [item.embedding for item in sorted(response.data, key=lambda d: d.index)]
            tokens += response.usage.total_tokens
        return Embedded(np.asarray(rows, dtype=np.float32), tokens)
