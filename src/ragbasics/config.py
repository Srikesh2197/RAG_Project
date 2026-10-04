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


def load_config[T: BaseModel](path: str | Path, model: type[T]) -> T:
    with open(path) as f:
        return model.model_validate(yaml.safe_load(f))
