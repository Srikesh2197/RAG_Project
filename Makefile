PYTHON ?= python3
VENV := .venv
BIN := $(VENV)/bin

.PHONY: setup data profile test lint lock

setup:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/python -m pip install --upgrade pip
	$(BIN)/python -m pip install -e ".[dev]"

data:
	$(BIN)/python scripts/download_data.py --config configs/data.yaml

profile:
	$(BIN)/python scripts/profile_data.py --config configs/data.yaml

test:
	$(BIN)/python -m pytest -q

lint:
	$(BIN)/python -m ruff check .

lock:
	$(BIN)/python -m pip freeze --exclude-editable > requirements.lock
