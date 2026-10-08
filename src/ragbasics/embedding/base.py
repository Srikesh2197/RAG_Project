"""What every embedder takes and returns, and the two things they share: truncating a
vector to fewer dimensions, and a disk cache.

`kind` says which side of the search a text is on. A question and the passage that
answers it are not paraphrases, so some models are trained with an instruction in front
of the query, some with one in front of both, and some with none. The caller says
"query" or "document"; each embedder knows what its model needs.
"""

import hashlib
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np

from ragbasics.embedding.cache import EmbeddingCache

Kind = Literal["document", "query"]


@dataclass(frozen=True)
class Embedded:
    vectors: np.ndarray  # shape (number of texts, dimensions), float32
    tokens: int  # tokens billed for this call


class Embedder(Protocol):
    model: str

    def embed(self, texts: list[str], kind: Kind = "document") -> Embedded: ...


def truncate(vectors: np.ndarray, dimensions: int | None) -> np.ndarray:
    """Keep the first `dimensions` numbers of each vector and scale it back to length 1.

    A model trained for this (Matryoshka training) packs most of the meaning into the
    leading dimensions, so the shortened vector still ranks well. The scaling is not
    optional: the first 256 numbers of a unit vector have a length below 1 that differs
    from vector to vector, and a dot product between them would no longer be a cosine.
    """
    if dimensions is None:
        return vectors
    if not 0 < dimensions <= vectors.shape[1]:
        raise ValueError(f"cannot truncate {vectors.shape[1]} dimensions to {dimensions}")
    cut = np.asarray(vectors[:, :dimensions], dtype=np.float32)
    norms = np.linalg.norm(cut, axis=1, keepdims=True)
    return np.divide(cut, norms, out=np.zeros_like(cut), where=norms > 0)


def cache_key(identity: str, text: str) -> str:
    return hashlib.sha256(identity.encode() + b"\x00" + text.encode()).hexdigest()


class CachingEmbedder:
    """Shared by the embedders that call a model: cache, then truncate.

    A subclass implements `_embed` (the model call, at full dimensions) and
    `cache_identity`. The cache holds full-size vectors, so trying another `dimensions`
    costs nothing, and the same text always gets the same vector. Without it, an API
    returns very slightly different numbers for the same text on different days.
    """

    model: str
    dimensions: int | None = None
    cache: EmbeddingCache | None = None

    def cache_identity(self, kind: Kind) -> str:
        """Everything besides the text that decides the vector: the model, and whatever
        is put in front of a text of this kind."""
        raise NotImplementedError

    def _embed(self, texts: list[str], kind: Kind) -> Embedded:
        raise NotImplementedError

    def uncached(self, texts: list[str], kind: Kind = "document") -> list[str]:
        """The texts `embed` would send to the model. Used to print a cost first."""
        if self.cache is None:
            return texts
        identity = self.cache_identity(kind)
        keys = {cache_key(identity, text): text for text in texts}
        found = self.cache.get_many(list(keys))
        return [text for key, text in keys.items() if key not in found]

    def embed(self, texts: list[str], kind: Kind = "document") -> Embedded:
        if self.cache is None:
            embedded = self._embed(texts, kind)
            return Embedded(truncate(embedded.vectors, self.dimensions), embedded.tokens)

        identity = self.cache_identity(kind)
        keys = [cache_key(identity, text) for text in texts]
        found = self.cache.get_many(keys)
        # Position of the first text for each key not cached: a text that appears twice
        # in one call is embedded once.
        first: dict[str, int] = {}
        for i, key in enumerate(keys):
            if key not in found:
                first.setdefault(key, i)
        missing = list(first.values())
        tokens = 0
        if missing:
            embedded = self._embed([texts[i] for i in missing], kind)
            tokens = embedded.tokens
            fresh = {keys[i]: row for i, row in zip(missing, embedded.vectors, strict=True)}
            self.cache.put_many(fresh)
            found |= fresh
        vectors = np.stack([found[key] for key in keys]).astype(np.float32)
        return Embedded(truncate(vectors, self.dimensions), tokens)
