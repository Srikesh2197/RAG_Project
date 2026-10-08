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
| 3 | Chunking | done | [03-chunking.md](docs/stages/03-chunking.md) |
| 4 | Embedding models | done | [04-embedding-models.md](docs/stages/04-embedding-models.md) |
| 5 | Sparse and hybrid retrieval | done | [05-sparse-hybrid-retrieval.md](docs/stages/05-sparse-hybrid-retrieval.md) |
| 6 | Reranking | next | |
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

To try another chunker (Stage 3), each config builds its own index:

```bash
.venv/bin/rag ingest --config configs/stage03_chunking/recursive-128.yaml --estimate
.venv/bin/rag ingest --config configs/stage03_chunking/recursive-128.yaml
.venv/bin/rag eval --config configs/stage03_chunking/recursive-128.yaml --retrieval-only
.venv/bin/python scripts/stage03_sweep.py   # the chunking table, against the baseline
```

To try another embedding model (Stage 4). The local models need `make setup-local` (PyTorch and sentence-transformers) and download their weights on first use:

```bash
.venv/bin/rag ingest --config configs/stage04_embedding/qwen3.yaml --estimate
.venv/bin/rag ingest --config configs/stage04_embedding/qwen3.yaml
.venv/bin/rag eval --config configs/stage04_embedding/qwen3.yaml --retrieval-only
.venv/bin/python scripts/stage04_sweep.py   # the embedding table, against the best config so far
```

To try keyword or hybrid retrieval (Stage 5). These rank the chunks of the `recursive-128` index, so nothing is embedded again; the BM25 index is built on first use:

```bash
.venv/bin/rag eval --config configs/stage05_retrieval/sparse.yaml --retrieval-only
.venv/bin/rag eval --config configs/stage05_retrieval/hybrid-rrf.yaml --retrieval-only
.venv/bin/python scripts/stage05_tune.py --estimate   # settings are chosen on the pool questions
.venv/bin/python scripts/stage05_sweep.py             # the retrieval table, against dense
```

`make app-offline` runs the same app with no keys, using a word-hashing embedder and no LLM. It shows the mechanics, not useful answers.

## Results

<!-- results:start -->
Percentages with 95% intervals from redrawing evidence clusters. "vs baseline" is the paired difference on the same questions; an interval that excludes 0 is a difference the question sample does not explain. "vs best so far" is the same difference against `retr-hybrid-rrf`, the configuration the current stage changes one variable in.

**Retrieval** (441 development questions with gold evidence)

| run | recall@5 | full support@5 | MRR@10 | nDCG@10 | recall in 2,000 tok | precision in 2,000 tok | recall in 2,000 tok vs baseline | precision in 2,000 tok vs baseline | recall in 2,000 tok vs best so far |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 48.2 (41–57) | 21.5 (15–32) | 59.0 (53–66) | 49.0 (43–56) | 44.8 (38–53) | 1.7 (1.5–1.9) |  |  | -28.1 (-32.7 to -23.5) |
| chunk-sentence-512 | 47.7 (41–56) | 20.6 (14–30) | 57.6 (53–64) | 48.8 (43–56) | 44.4 (38–53) | 1.7 (1.5–2.0) | -0.4 (-2.7 to +1.6) | +0.03 (-0.06 to +0.12) | -28.5 (-33.0 to -24.2) |
| chunk-recursive-512 | 44.8 (39–52) | 17.5 (12–25) | 54.1 (50–60) | 46.4 (41–53) | 42.4 (37–50) | 1.6 (1.4–1.8) | -2.4 (-5.2 to -0.2) | -0.06 (-0.17 to +0.04) | -30.5 (-35.4 to -25.8) |
| chunk-recursive-128 | 35.2 (30–42) | 12.0 (8–18) | 45.1 (40–53) | 37.4 (33–44) | 63.8 (59–71) | 2.5 (2.2–2.7) | +19.0 (+14.7 to +22.7) | +0.81 (+0.54 to +1.09) | -9.1 (-12.5 to -6.1) |
| chunk-recursive-256 | 39.1 (33–47) | 15.2 (10–23) | 46.4 (42–53) | 39.9 (35–47) | 51.2 (44–60) | 1.9 (1.7–2.2) | +6.4 (+2.7 to +10.9) | +0.24 (+0.09 to +0.41) | -21.7 (-25.4 to -17.5) |
| chunk-recursive-1024 | 53.6 (46–61) | 25.2 (18–35) | 61.6 (56–68) | 54.9 (49–61) | 36.7 (30–44) | 1.4 (1.3–1.6) | -8.1 (-13.2 to -4.1) | -0.28 (-0.45 to -0.14) | -36.3 (-42.6 to -30.3) |
| chunk-fixed-128 | 38.2 (33–45) | 13.6 (9–21) | 47.7 (43–54) | 37.0 (33–43) | 60.9 (55–68) | 2.2 (2.0–2.4) | +16.1 (+12.3 to +19.3) | +0.50 (+0.31 to +0.66) | -12.0 (-16.4 to -7.7) |
| chunk-fixed-512-overlap-10 | 46.3 (40–56) | 20.6 (14–30) | 52.2 (47–59) | 44.3 (38–52) | 42.0 (36–51) | 1.6 (1.4–1.9) | -2.8 (-7.3 to +0.7) | -0.07 (-0.24 to +0.09) | -31.0 (-36.2 to -25.9) |
| chunk-fixed-512-overlap-20 | 44.0 (38–52) | 17.7 (12–27) | 51.6 (46–58) | 43.8 (39–51) | 38.8 (33–47) | 1.5 (1.3–1.7) | -6.0 (-10.1 to -2.6) | -0.20 (-0.36 to -0.05) | -34.2 (-39.3 to -29.3) |
| chunk-parent-child-128-512 | 50.7 (45–58) | 22.9 (17–32) | 59.9 (56–66) | 52.6 (48–60) | 45.7 (40–54) | 1.8 (1.6–2.0) | +0.9 (-3.2 to +5.0) | +0.10 (-0.07 to +0.26) | -27.2 (-31.1 to -23.5) |
| chunk-recursive-64 | 37.8 (33–44) | 13.6 (9–20) | 47.9 (42–55) | 38.9 (35–45) | 73.2 (69–80) | 3.0 (2.6–3.4) | +28.4 (+23.8 to +32.7) | +1.27 (+0.85 to +1.82) | +0.3 (-4.2 to +4.3) |
| chunk-parent-child-128-256 | 42.3 (36–50) | 16.1 (11–24) | 52.0 (46–60) | 44.0 (39–51) | 55.0 (49–63) | 2.1 (1.9–2.3) | +10.2 (+5.3 to +14.4) | +0.43 (+0.23 to +0.62) | -18.0 (-21.9 to -14.4) |
| chunk-semantic-128 | 38.8 (33–46) | 15.0 (10–21) | 48.3 (43–55) | 39.9 (35–46) | 63.4 (57–71) | 2.4 (2.3–2.6) | +18.6 (+14.5 to +22.4) | +0.75 (+0.52 to +0.92) | -9.6 (-13.6 to -5.4) |
| chunk-recursive-128-k25 | 35.2 (30–42) | 12.0 (8–18) | 45.2 (40–53) | 37.4 (33–44) | 63.8 (59–71) | 2.5 (2.2–2.7) | +19.0 (+14.7 to +22.7) | +0.81 (+0.54 to +1.09) | -9.1 (-12.5 to -6.1) |
| embed-3-small-512 | 33.4 (28–41) | 10.7 (7–16) | 43.2 (38–52) | 35.8 (31–43) | 60.5 (55–67) | 2.4 (2.1–2.6) | +15.7 (+10.7 to +20.0) | +0.68 (+0.39 to +0.98) | -12.4 (-16.5 to -9.0) |
| embed-3-small-256 | 31.0 (26–38) | 10.2 (6–16) | 40.1 (35–46) | 32.9 (29–38) | 56.7 (51–63) | 2.2 (2.0–2.5) | +11.9 (+6.5 to +16.1) | +0.56 (+0.27 to +0.82) | -16.2 (-21.3 to -12.3) |
| embed-3-large-1024 | 37.2 (31–45) | 14.3 (9–22) | 48.1 (43–56) | 39.1 (33–47) | 62.1 (53–72) | 2.4 (2.2–2.7) | +17.3 (+12.7 to +22.7) | +0.69 (+0.53 to +0.86) | -10.9 (-16.8 to -5.9) |
| embed-3-large-256 | 30.3 (26–37) | 8.4 (5–13) | 40.7 (36–47) | 32.5 (28–39) | 54.8 (47–63) | 2.1 (1.9–2.4) | +10.0 (+4.9 to +15.0) | +0.44 (+0.24 to +0.62) | -18.1 (-23.7 to -13.0) |
| embed-bge-small | 33.3 (27–41) | 13.2 (9–19) | 39.4 (34–48) | 33.0 (28–40) | 53.9 (47–62) | 2.1 (1.9–2.3) | +9.1 (+5.1 to +13.0) | +0.38 (+0.22 to +0.52) | -19.0 (-23.2 to -14.6) |
| embed-bge-small-noprefix | 31.5 (25–39) | 12.5 (9–18) | 37.2 (32–45) | 31.0 (26–38) | 51.5 (45–59) | 2.0 (1.8–2.2) | +6.7 (+2.0 to +10.4) | +0.29 (+0.09 to +0.43) | -21.4 (-25.8 to -17.6) |
| embed-bge-small-96 | 21.1 (17–27) | 7.0 (4–12) | 25.0 (21–31) | 21.7 (18–27) | 40.5 (34–48) | 1.5 (1.4–1.7) | -4.3 (-9.3 to -0.2) | -0.15 (-0.34 to +0.02) | -32.4 (-37.3 to -28.0) |
| embed-qwen3 | 42.1 (34–52) | 19.5 (14–28) | 51.8 (45–62) | 42.1 (36–51) | 64.1 (56–74) | 2.5 (2.2–2.7) | +19.3 (+15.5 to +23.5) | +0.78 (+0.60 to +0.95) | -8.9 (-13.4 to -4.1) |
| embed-qwen3-noprefix | 40.0 (33–50) | 18.4 (13–26) | 49.6 (42–60) | 40.0 (33–49) | 61.0 (52–72) | 2.3 (2.1–2.7) | +16.2 (+12.0 to +22.1) | +0.66 (+0.50 to +0.87) | -11.9 (-17.3 to -6.8) |
| embed-qwen3-256 | 36.4 (30–45) | 13.6 (9–20) | 46.3 (39–56) | 36.9 (31–46) | 55.1 (48–65) | 2.1 (1.9–2.4) | +10.3 (+6.2 to +15.4) | +0.45 (+0.27 to +0.63) | -17.9 (-21.9 to -13.4) |
| embed-3-large | 38.7 (32–47) | 15.9 (11–24) | 49.2 (44–57) | 40.8 (35–49) | 63.8 (55–74) | 2.4 (2.2–2.7) | +19.0 (+14.4 to +24.6) | +0.76 (+0.60 to +0.92) | -9.1 (-15.1 to -4.1) |
| retr-sparse-default | 46.6 (39–56) | 20.4 (14–30) | 59.5 (52–69) | 48.4 (42–57) | 65.9 (60–75) | 2.6 (2.3–2.9) | +21.1 (+16.2 to +26.0) | +0.93 (+0.60 to +1.26) | -7.0 (-9.5 to -4.1) |
| retr-sparse | 47.8 (41–58) | 21.5 (15–31) | 60.1 (53–69) | 49.2 (43–58) | 68.2 (62–78) | 2.7 (2.3–3.1) | +23.4 (+17.5 to +29.5) | +1.04 (+0.65 to +1.48) | -4.7 (-7.7 to -1.4) |
| retr-hybrid-rrf | 48.8 (42–58) | 22.9 (16–32) | 57.2 (51–65) | 48.6 (43–57) | 72.9 (68–81) | 2.9 (2.6–3.4) | +28.1 (+23.5 to +32.7) | +1.26 (+0.87 to +1.74) |  |
| retr-hybrid-rrf-k10 | 49.1 (43–58) | 23.4 (17–33) | 58.8 (53–67) | 50.6 (45–59) | 72.9 (68–81) | 2.9 (2.6–3.3) | +28.1 (+23.4 to +32.6) | +1.22 (+0.86 to +1.65) | -0.1 (-1.3 to +1.3) |
| retr-hybrid-weighted | 50.4 (44–60) | 24.7 (18–35) | 63.7 (58–72) | 53.0 (47–61) | 73.3 (68–81) | 2.9 (2.6–3.3) | +28.5 (+24.1 to +33.1) | +1.25 (+0.88 to +1.69) | +0.4 (-0.7 to +1.6) |
| retr-sparse-header | 50.9 (43–61) | 24.9 (17–36) | 65.3 (58–74) | 53.6 (47–62) | 74.3 (68–82) | 2.9 (2.7–3.2) | +29.5 (+25.1 to +34.1) | +1.26 (+0.95 to +1.60) | +1.3 (-1.0 to +4.2) |
| retr-hybrid-rrf-k10-header | 52.6 (46–62) | 27.2 (20–37) | 61.6 (56–69) | 52.9 (48–61) | 76.3 (71–84) | 3.0 (2.7–3.4) | +31.5 (+26.9 to +35.7) | +1.34 (+0.99 to +1.76) | +3.4 (+1.5 to +5.3) |
| retr-hybrid-rrf-top25 | 48.8 (42–58) | 22.9 (16–32) | 57.2 (51–65) | 48.6 (43–57) | 72.9 (68–81) | 2.9 (2.6–3.4) | +28.1 (+23.5 to +32.7) | +1.26 (+0.87 to +1.74) | +0.0 (+0.0 to +0.0) |

**Answers**

| run | n | correct | correct, answerable | abstains on null | false abstentions | faithful | correct vs baseline | cost |
|---|---|---|---|---|---|---|---|---|
| baseline | 150 | 51.3 (42–60) | 44.7 (33–54) | 100.0 (100–100) | 52.3 (43–63) | 81.0 (68–90) |  | $1.31 |
| baseline-r1 | 150 | 50.0 (41–58) | 43.2 (32–52) | 100.0 (100–100) | 53.0 (44–64) | 82.3 (73–90) | -1.3 (-4 to +2) | $1.31 |
| baseline-gold | 150 | 46.0 (36–55) | 38.6 (25–49) | 100.0 (100–100) | 61.4 (51–75) | 58.8 (41–75) | -5.3 (-14 to +2) | $0.17 |
| baseline-closed_book | 150 | 47.3 (36–56) | 41.7 (28–52) | 88.9 (72–100) | 56.1 (46–69) |  | -4.0 (-14 to +4) | $0.14 |
| guesser | 150 | 52.7 (40–67) | 46.2 (32–63) | 100.0 (100–100) | 0.0 (0.0–0.0) |  | +1.3 (-13 to +17) |  |
| chunk-recursive-128-k25 | 150 | 56.0 (47–64) | 50.0 (39–59) | 100.0 (100–100) | 47.0 (39–58) | 68.6 (57–78) | +4.7 (-2 to +11) | $1.43 |
| retr-hybrid-rrf-top25 | 150 | 65.3 (58–73) | 60.6 (52–69) | 100.0 (100–100) | 38.6 (30–47) | 74.1 (63–82) | +14.0 (+7 to +23) | $1.43 |

**Answer correctness by question type**

| run | comparison | inference | temporal | null |
|---|---|---|---|---|
| baseline | 31.7 (22–43) | 84.4 (61–94) | 32.4 (17–48) | 100.0 (100–100) |
| baseline-r1 | 28.6 (18–40) | 84.4 (67–96) | 32.4 (17–48) | 100.0 (100–100) |
| baseline-gold | 28.6 (18–40) | 96.9 (86–100) | 5.4 (0–11) | 100.0 (100–100) |
| baseline-closed_book | 15.9 (7–26) | 96.9 (86–100) | 37.8 (20–55) | 88.9 (72–100) |
| guesser | 54.0 (42–66) | 34.4 (0–72) | 43.2 (32–59) | 100.0 (100–100) |
| chunk-recursive-128-k25 | 33.3 (22–47) | 100.0 (100–100) | 35.1 (20–53) | 100.0 (100–100) |
| retr-hybrid-rrf-top25 | 58.7 (45–74) | 100.0 (100–100) | 29.7 (16–44) | 100.0 (100–100) |
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
