from pathlib import Path

import pytest
from pydantic import ValidationError

from ragbasics.config import ComponentSpec, DataConfig, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_repo_data_config_loads():
    cfg = load_config(REPO_ROOT / "configs" / "data.yaml", DataConfig)
    assert cfg.dev_size == 500
    assert cfg.processed_dir == Path("data/processed")


def test_unknown_key_is_rejected(tmp_path):
    path = tmp_path / "component.yaml"
    path.write_text("name: fixed\nparms: {size: 512}\n")
    with pytest.raises(ValidationError):
        load_config(path, ComponentSpec)


def test_component_params_default_to_empty(tmp_path):
    path = tmp_path / "component.yaml"
    path.write_text("name: fixed\n")
    assert load_config(path, ComponentSpec) == ComponentSpec(name="fixed", params={})
