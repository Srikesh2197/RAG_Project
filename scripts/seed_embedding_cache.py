"""Copy the vectors of an index that already exists into the embedding cache.

    python scripts/seed_embedding_cache.py configs/stage03_chunking/recursive-128.yaml

The Stage 3 indexes were built before the cache existed. Seeding it from one of them
lets a config with the same chunker and model, and a shorter `dimensions`, build its
index without paying to embed the corpus again. Needs no API key.
"""

import argparse
from pathlib import Path

from ragbasics.config import PipelineConfig, load_config
from ragbasics.embedding.base import cache_key
from ragbasics.embedding.cache import EmbeddingCache
from ragbasics.pipeline import Pipeline

CACHE = Path("data/cache/embeddings.sqlite")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--cache", type=Path, default=CACHE)
    args = parser.parse_args()

    pipeline = Pipeline(load_config(args.config, PipelineConfig))
    if not pipeline.load():
        raise SystemExit(f"No index in {pipeline.index_dir}.")
    if getattr(pipeline.embedder, "dimensions", None):
        raise SystemExit("This index holds truncated vectors; the cache holds full ones.")
    identity = pipeline.embedder.cache_identity("document")
    cache = EmbeddingCache(args.cache)
    cache.put_many(
        {
            cache_key(identity, chunk.text_to_embed): vector
            for chunk, vector in zip(pipeline.store.chunks, pipeline.store.vectors, strict=True)
        }
    )
    print(f"Seeded {len(pipeline.store)} vectors for {identity} into {args.cache}.")


if __name__ == "__main__":
    main()
