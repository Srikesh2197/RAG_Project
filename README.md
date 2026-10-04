# RAG Basics

A retrieval-augmented generation (RAG) pipeline built by hand, one component at a time, with every design choice measured against a naive baseline.

The project starts from a deliberately simple pipeline, adds an evaluation harness, then changes one variable per stage: chunking, embedding model, hybrid search, reranking, query transformation, context assembly, citations, parsing and indexing. Each stage ends with a write-up of the concept, the options, the choice, the measured result and when a production system would choose differently.

Core mechanics are written from scratch first. Library equivalents are shown next to them with a test that both give the same output.

## Stages

| # | Stage | Status | Write-up |
|---|---|---|---|
| 0 | Setup and data | done | [00-setup.md](docs/stages/00-setup.md) |
| 1 | Naive baseline and Streamlit app | next | |
| 2 | Evaluation harness | | |
| 3 | Chunking | | |
| 4 | Embedding models | | |
| 5 | Sparse and hybrid retrieval | | |
| 6 | Reranking | | |
| 7 | Query transformation | | |
| 8 | Context assembly and prompt construction | | |
| 9 | Citations and abstention | | |
| 10 | PDF parsing and ingestion | | |
| 11 | Indexing, vector stores and metadata filtering | | |
| 12 | Capstone: ablation, held-out test | | |
| A1–A4 | Contextual retrieval, agentic loop, guardrails, multimodal | | |

The survey of the RAG landscape that shaped this plan is in [docs/research-notes.md](docs/research-notes.md).

## Quickstart

Requires Python 3.12 or newer.

```bash
make setup    # create .venv and install
make data     # download MultiHop-RAG at a pinned revision, build gold spans and splits
make test
make profile  # write results/stage00_data_profile.md
```

## Data

The corpus is [MultiHop-RAG](https://huggingface.co/datasets/yixuantt/MultiHopRAG) (Tang and Yang, 2024; ODC-BY licence): 609 news articles and 2,556 questions with reference answers and gold evidence sentences. It is downloaded by `make data` and not stored in this repo.

Gold evidence is stored as character spans in the unchunked articles, so the labels stay valid for any chunking strategy. The development and test splits never share a gold sentence. Details are in the [Stage 0 write-up](docs/stages/00-setup.md).

## Layout

```
configs/      YAML configs; a stage comparison is a config diff
src/ragbasics pipeline code, one package per component
scripts/      data download and profiling
tests/        unit tests; no API keys or downloads needed
docs/         stage write-ups and research notes
results/      committed result tables
```
