"""Embedding models run on this machine through sentence-transformers.

The library loads the weights and runs the forward pass. What stays in this file, where
it can be read, is everything that differs between models and is easy to get wrong:

- `query_prefix` and `document_prefix` are plain strings put in front of the text. The
  config file spells them out, so the prefix a model was trained with is visible and
  can be switched off to measure what it is worth.
- Vectors come back as the model produced them. Not every model returns length 1; the
  store normalises, and `dimensions` truncates and re-normalises.
- `max_tokens` is the most the model reads. Longer inputs are cut without any error, so
  the embedder counts them in `truncated_inputs`.

sentence-transformers and PyTorch are in the "local" extra and are imported on first
use, so the rest of the project and its tests run without them.
"""

import threading
from typing import Any

import numpy as np

from ragbasics.embedding.base import CachingEmbedder, Embedded, Kind
from ragbasics.registry import register


@register("embedder", "local")
class LocalEmbedder(CachingEmbedder):
    def __init__(
        self,
        model: str,
        revision: str | None = None,
        query_prefix: str = "",
        document_prefix: str = "",
        dimensions: int | None = None,
        max_tokens: int | None = None,
        batch_size: int = 32,
        device: str | None = None,
        encoder: Any = None,
    ):
        self.model = model
        self.revision = revision
        self.prefixes = {"query": query_prefix, "document": document_prefix}
        self.dimensions = dimensions
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.device = device
        self._encoder = encoder
        self.truncated_inputs = 0  # inputs longer than the model reads
        # The evaluator embeds questions from several threads; one forward pass at a time.
        self._lock = threading.Lock()

    @property
    def encoder(self) -> Any:
        # Loaded on first use: building a pipeline needs neither the library nor the weights.
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(
                self.model, revision=self.revision, device=self.device
            )
            if self.max_tokens:
                self._encoder.max_seq_length = self.max_tokens
        return self._encoder

    def cache_identity(self, kind: Kind) -> str:
        return (
            f"local:{self.model}@{self.revision}:{self.max_tokens}:{self.prefixes[kind]}"
        )

    def _embed(self, texts: list[str], kind: Kind) -> Embedded:
        inputs = [self.prefixes[kind] + text for text in texts]
        with self._lock:
            limit = self.encoder.max_seq_length
            if limit:
                lengths = self.encoder.tokenizer(inputs, truncation=False)["input_ids"]
                self.truncated_inputs += sum(len(ids) > limit for ids in lengths)
            vectors = self.encoder.encode(
                inputs,
                batch_size=self.batch_size,
                normalize_embeddings=False,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        # Nothing is billed for a local model.
        return Embedded(np.asarray(vectors, dtype=np.float32), tokens=0)
