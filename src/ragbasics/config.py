"""Typed configs loaded from YAML.

Every model forbids unknown keys, so a typo in a config file fails at load time
instead of silently running the default.
"""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ComponentSpec(StrictModel):
    """One swappable component: a registry name plus its constructor parameters."""

    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class DataConfig(StrictModel):
    repo_id: str
    revision: str
    corpus_file: str
    questions_file: str
    raw_dir: Path
    processed_dir: Path
    dev_size: int = Field(gt=0)
    test_size: int = Field(gt=0)
    answer_subset_size: int = Field(gt=0)
    max_cluster_share: float = Field(gt=0, le=1)
    seed: int


class PipelineConfig(StrictModel):
    """One pipeline: a component per slot. A stage comparison is a diff of two of these."""

    name: str
    index_dir: Path
    chunker: ComponentSpec
    embedder: ComponentSpec
    store: ComponentSpec
    top_k: int = Field(gt=0)
    # Search this many times deeper than asked, before chunks that return the same text
    # are collapsed into one. Parent-child chunking needs it; flat chunking leaves it at 1.
    overfetch: int = Field(default=1, gt=0)
    # A SQLite file of vectors already computed. With it, the same text always gets the
    # same vector, and a truncated length reuses the full one. Not part of what the index
    # depends on, so adding it to a config does not invalidate an index.
    embedding_cache: Path | None = None
    generator: ComponentSpec


class BootstrapConfig(StrictModel):
    samples: int = Field(gt=0)
    seed: int


class EvalConfig(StrictModel):
    """How a pipeline is evaluated. Shared by every run, so runs stay comparable."""

    questions: Path
    documents: Path
    depth: int = Field(gt=0)  # chunks retrieved per question for the retrieval metrics
    ks: list[int]  # cut-offs for recall@k and full-support@k
    rank_k: int = Field(gt=0)  # cut-off for MRR and nDCG
    token_budget: int = Field(gt=0)
    judge: ComponentSpec
    bootstrap: BootstrapConfig
    workers: int = Field(gt=0)  # parallel API calls
    cache: Path
    confirm_above_usd: float  # a run estimated above this needs --yes


def load_config[T: BaseModel](path: str | Path, model: type[T]) -> T:
    with open(path) as f:
        return model.model_validate(yaml.safe_load(f))
