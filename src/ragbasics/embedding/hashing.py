"""An embedder that needs no model and no API key.

Each word is hashed to one of `dimensions` slots, and a text's vector is its word counts
in those slots. Two texts score as similar only when they share words, so this is
keyword matching in vector form. It exists so the tests, CI and an offline demo can run
the whole pipeline; it is not a baseline for any measurement.
"""

import re
import zlib

import numpy as np

from ragbasics.embedding.base import Embedded
from ragbasics.registry import register

WORD = re.compile(r"\w+")


@register("embedder", "hashing")
class HashingEmbedder:
    def __init__(self, dimensions: int = 512):
        self.dimensions = dimensions
        self.model = f"hashing-{dimensions}"

    def embed(self, texts: list[str]) -> Embedded:
        vectors = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in WORD.findall(text.lower()):
                vectors[row, zlib.crc32(word.encode()) % self.dimensions] += 1.0
        return Embedded(vectors, tokens=0)
