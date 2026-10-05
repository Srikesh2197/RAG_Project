"""What every embedder returns."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class Embedded:
    vectors: np.ndarray  # shape (number of texts, dimensions), float32
    tokens: int  # tokens billed for this call


class Embedder(Protocol):
    model: str

    def embed(self, texts: list[str]) -> Embedded: ...
