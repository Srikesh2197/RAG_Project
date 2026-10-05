PYTHON ?= python3
VENV := .venv
BIN := $(VENV)/bin

.PHONY: setup data profile ingest app app-offline test lint lock

setup:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/python -m pip install --upgrade pip
	$(BIN)/python -m pip install -e ".[dev]"

data:
	$(BIN)/python scripts/download_data.py --config configs/data.yaml

profile:
	$(BIN)/python scripts/profile_data.py --config configs/data.yaml

# Stage 1. `ingest` and `app` need OPENAI_API_KEY and ANTHROPIC_API_KEY in .env.
ingest:
	$(BIN)/rag ingest --config configs/baseline.yaml

app:
	$(BIN)/streamlit run app/streamlit_app.py -- --config configs/baseline.yaml

# No keys needed: word-hashing embedder, no LLM.
app-offline:
	$(BIN)/rag ingest --config configs/offline.yaml
	$(BIN)/streamlit run app/streamlit_app.py -- --config configs/offline.yaml

test:
	$(BIN)/python -m pytest -q

lint:
	$(BIN)/python -m ruff check .

lock:
	$(BIN)/python -m pip freeze --exclude-editable > requirements.lock
