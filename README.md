# RAG Basics

A retrieval-augmented generation (RAG) pipeline built by hand, one component at a time, with every design choice measured against a naive baseline.

The project starts from a deliberately simple pipeline, adds an evaluation harness, then changes one variable per stage: chunking, embedding model, hybrid search, reranking, query transformation, context assembly, citations, parsing and indexing. Each stage ends with a write-up of the concept, the options, the choice, the measured result and when a production system would choose differently.

Core mechanics are written from scratch first. Library equivalents are shown next to them with a test that both give the same output.

## Stages

| # | Stage | Status | Write-up |
|---|---|---|---|
| 0 | Setup and data | done | [00-setup.md](docs/stages/00-setup.md) |
| 1 | Naive baseline and Streamlit app | done | [01-naive-baseline.md](docs/stages/01-naive-baseline.md) |
| 2 | Evaluation harness | done; human check of the judge deferred | [02-evaluation-harness.md](docs/stages/02-evaluation-harness.md) |
| 3 | Chunking | next | |
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

To run the pipeline, copy `.env.example` to `.env` and add `OPENAI_API_KEY` and `ANTHROPIC_API_KEY`.

```bash
.venv/bin/rag check   # confirm the keys and model ids
make ingest           # chunk and embed the corpus, about 3 cents
make app              # Streamlit app: ask questions, see retrieved chunks, scores and sources
.venv/bin/rag ask "Which company removed AI-created songs and added an AI DJ?"
```

To score a pipeline on the development questions:

```bash
.venv/bin/rag eval --retrieval-only   # retrieval metrics for all 500 questions, under 1 cent
.venv/bin/rag eval --estimate         # print the cost of a full run and stop
.venv/bin/rag eval --yes              # answers and judge on the 150-question subset
.venv/bin/rag eval --mode gold        # reference rows: gold, closed_book, guesser
.venv/bin/rag compare baseline-gold baseline
.venv/bin/rag report                  # rebuild results/results.csv and the table below
```

`make app-offline` runs the same app with no keys, using a word-hashing embedder and no LLM. It shows the mechanics, not useful answers.

## Results

<!-- results:start -->
Percentages with 95% intervals from redrawing evidence clusters. "vs baseline" is the paired difference on the same questions; an interval that excludes 0 is a difference the question sample does not explain.

**Retrieval** (441 development questions with gold evidence)

| run | recall@5 | full support@5 | MRR@10 | nDCG@10 | recall in 2,000 tok | precision in 2,000 tok |
|---|---|---|---|---|---|---|
| baseline | 48.2 (41–57) | 21.5 (15–32) | 59.0 (53–66) | 49.0 (43–56) | 44.8 (38–53) | 1.7 (1.5–1.9) |

**Answers**

| run | n | correct | correct, answerable | abstains on null | false abstentions | faithful | correct vs baseline | cost |
|---|---|---|---|---|---|---|---|---|
| baseline | 150 | 51.3 (42–60) | 44.7 (33–54) | 100.0 (100–100) | 52.3 (43–63) | 81.0 (68–90) |  | $1.31 |
| baseline-r1 | 150 | 50.0 (41–58) | 43.2 (32–52) | 100.0 (100–100) | 53.0 (44–64) | 82.3 (73–90) | -1.3 (-4 to +2) | $1.31 |
| baseline-gold | 150 | 46.0 (36–55) | 38.6 (25–49) | 100.0 (100–100) | 61.4 (51–75) | 58.8 (41–75) | -5.3 (-14 to +2) | $0.17 |
| baseline-closed_book | 150 | 47.3 (36–56) | 41.7 (28–52) | 88.9 (72–100) | 56.1 (46–69) |  | -4.0 (-14 to +4) | $0.14 |
| guesser | 150 | 52.7 (40–67) | 46.2 (32–63) | 100.0 (100–100) | 0.0 (0.0–0.0) |  | +1.3 (-13 to +17) |  |

**Answer correctness by question type**

| run | comparison | inference | temporal | null |
|---|---|---|---|---|
| baseline | 31.7 (22–43) | 84.4 (61–94) | 32.4 (17–48) | 100.0 (100–100) |
| baseline-r1 | 28.6 (18–40) | 84.4 (67–96) | 32.4 (17–48) | 100.0 (100–100) |
| baseline-gold | 28.6 (18–40) | 96.9 (86–100) | 5.4 (0–11) | 100.0 (100–100) |
| baseline-closed_book | 15.9 (7–26) | 96.9 (86–100) | 37.8 (20–55) | 88.9 (72–100) |
| guesser | 54.0 (42–66) | 34.4 (0–72) | 43.2 (32–59) | 100.0 (100–100) |
<!-- results:end -->

## Data

The corpus is [MultiHop-RAG](https://huggingface.co/datasets/yixuantt/MultiHopRAG) (Tang and Yang, 2024; ODC-BY licence): 609 news articles and 2,556 questions with reference answers and gold evidence sentences. It is downloaded by `make data` and not stored in this repo.

Gold evidence is stored as character spans in the unchunked articles, so the labels stay valid for any chunking strategy. The development and test splits never share a gold sentence. Details are in the [Stage 0 write-up](docs/stages/00-setup.md).

## Layout

```
configs/      YAML configs; a stage comparison is a config diff
src/ragbasics pipeline code, one package per component
app/          Streamlit app
scripts/      data download and profiling
tests/        unit tests; no API keys or downloads needed
framework_equivalents/ library versions of hand-written parts
docs/         stage write-ups and research notes
results/      committed result tables
```
