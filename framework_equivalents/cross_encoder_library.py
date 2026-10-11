"""Library equivalent of the hand-written cross-encoder reranker: sentence-transformers.

    pip install -e ".[local]"

`CrossEncoder.predict` does what `ragbasics.rerank.cross_encoder` spells out: it pairs
each passage with the question, tokenises the pair with truncation, batches (sorted by
length), runs the model and returns one score per pair. `CrossEncoder.rank` then sorts.

What differs:

- The library applies an activation to the model's output. Which one is stored in the
  model's own config: none for the ms-marco models, a sigmoid for bge-reranker-v2-m3. A
  sigmoid changes no ranking. The hand-written reranker keeps the raw number, so the
  comparison here asks the library for the raw number too.
- The library truncates a long pair without saying so. The hand-written reranker counts
  the pairs it had to cut.
- The library has no score cache.
"""

from sentence_transformers import CrossEncoder
from torch import nn


def load(model: str, revision: str | None = None, max_length: int = 512) -> CrossEncoder:
    return CrossEncoder(model, revision=revision, max_length=max_length)


def scores(encoder: CrossEncoder, pairs: list[tuple[str, str]]) -> list[float]:
    """The model's raw output for each (question, passage) pair."""
    return encoder.predict(pairs, activation_fn=nn.Identity(), show_progress_bar=False).tolist()


def rank(encoder: CrossEncoder, question: str, passages: list[str]) -> list[int]:
    """Passage numbers, best first."""
    return [hit["corpus_id"] for hit in encoder.rank(question, passages)]
