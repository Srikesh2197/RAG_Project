"""OpenAI embedding models."""

from typing import Any

import numpy as np

from ragbasics.embedding.base import Embedded
from ragbasics.registry import register


@register("embedder", "openai")
class OpenAIEmbedder:
    def __init__(
        self, model: str = "text-embedding-3-small", batch_size: int = 100, client: Any = None
    ):
        self.model = model
        self.batch_size = batch_size
        self._client = client

    @property
    def client(self) -> Any:
        # Created on first use, so building a pipeline does not need an API key.
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI()
        return self._client

    def embed(self, texts: list[str]) -> Embedded:
        rows: list[list[float]] = []
        tokens = 0
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            response = self.client.embeddings.create(model=self.model, input=batch)
            rows += [item.embedding for item in sorted(response.data, key=lambda d: d.index)]
            tokens += response.usage.total_tokens
        return Embedded(np.asarray(rows, dtype=np.float32), tokens)
