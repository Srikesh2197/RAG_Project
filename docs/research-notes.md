# RAG landscape notes

Survey date: 1 October 2026. This is the research behind the project plan: what the options are at each step of a RAG pipeline, what practitioners currently default to, and what evidence exists.

How to read it:

- **[I]** marks an independent or academic source. **[V]** marks a vendor reporting on its own product or a competitor's.
- Most figures were read through a summarising fetch tool and have not been re-checked against the original page. Re-check a number before quoting it in a stage write-up.
- "Not confirmed" lists at the end of each section are things the survey could not verify.

## 1. Document parsing

| Tool | Runs locally | Licence | Notes |
|---|---|---|---|
| pypdf | yes | BSD | Text only. No tables or layout. |
| PyMuPDF / pymupdf4llm | yes | AGPL-3.0 or paid | Fast; Markdown output; tables as Markdown or HTML. |
| Docling | yes | MIT | PDF, DOCX, PPTX, HTML; layout and table models. |
| Marker | yes | Code Apache-2.0; model weights restricted above $5M revenue | Runs on CPU, GPU or Apple MPS. |
| MinerU | yes | Its own Apache-based licence with extra conditions | Strong on formulas and scientific PDFs. |
| Unstructured (open source) | yes | Apache-2.0 | `fast` and `hi_res` strategies. |
| LlamaParse | cloud | Proprietary | Priced in credits per page. |
| Vision LLMs, Mistral OCR | cloud | Proprietary | Highest quality on hard pages. |

**Default.** Plain extraction (pypdf or pymupdf4llm) for born-digital PDFs, Docling when tables and layout matter, and a vision model only for hard or scanned pages.

**Evidence.** Parser benchmarks disagree badly, because each uses different documents and metrics.

- [I] A small independent test (40 documents) scored vision models at 70–75% overall and Docling at 38%, with Docling at 64% on tables.
- [V] A vendor test on 500 PDFs scored Docling at 97.9% on tables.
- [V] Unstructured's own benchmark scores itself first on tables and Docling at 0.657.
- [V] Marker reports 76.0 on a 1,403-PDF benchmark on GPU; the independent subset above gave it 55.2.

**For this project.** Stage 10 measures parsers on our own PDFs. PyMuPDF's AGPL licence matters for a public repo, so it is an optional extra.

**Not confirmed:** MinerU's current version and licence history; pypdf and Unstructured licences were stated from memory.

Sources:
- https://github.com/ukanwat/ocr4-structure-benchmark (2026)
- https://github.com/datalab-to/marker
- https://www.ertas.ai/blog/pdf-parsing-accuracy-benchmark-docling-unstructured (March 2026)
- https://unstructured.io/benchmarks
- https://github.com/docling-project/docling
- https://pypi.org/project/pymupdf4llm/

## 2. Chunking

**Options.** Fixed-size; recursive (split on paragraph, then sentence, then word boundaries); sentence-based; structure-aware (headings, Markdown); semantic (split where consecutive sentence embeddings diverge); LLM-guided; parent-child (search small chunks, return the larger block around them); late chunking (embed the whole document, then pool per chunk); contextual enrichment (an LLM writes a context line per chunk); propositions (an LLM rewrites text into standalone facts).

**Default.** Recursive or structure-aware splitting at about 256–512 tokens with 0–20% overlap. Semantic chunking is not the default.

**Evidence.**

- [I] Qu, Tu and Bao (NAACL Findings 2025): the cost of semantic chunking is not justified by consistent gains over fixed-size chunks.
- [I] A nine-dataset comparison (May 2026) measured accuracy@5 of 89.4% for recursive-semantic, 87.7% for fixed-size (under a second of chunking time), 86.9% for a graph method (3 hours) and 85.4% for an LLM chunker (8 hours).
- [I] Zhou et al. (Feb 2026): the best strategy depends on the task. Structure-based methods win for retrieval across a corpus; an LLM chunker wins for retrieval inside one document.
- [V] Chroma (July 2024): recursive chunks of 200–400 tokens with no overlap reached 88–90% token-level recall; semantic and LLM chunkers reached 91–92%. Overlap lowered precision.
- [I] Merola and Singh (2025): contextual retrieval with rank fusion and reranking was the most accurate and the most expensive. Late chunking was cheaper and inconsistent.

**For this project.** Stage 3 tests one semantic configuration against recursive splitting and measures chunking time alongside retrieval quality.

**Not confirmed:** a widely repeated 2026 benchmark claiming semantic chunking scored 54% against 69% for recursive; the primary source could not be found.

Sources:
- https://aclanthology.org/2025.findings-naacl.114/ (April 2025)
- https://arxiv.org/abs/2606.00881 (May 2026)
- https://arxiv.org/abs/2602.16974 (February 2026)
- https://www.trychroma.com/research/evaluating-chunking (July 2024)
- https://arxiv.org/abs/2504.19754 (April 2025)

## 3. Embedding models

| Model | Dimensions | Max tokens | $ per 1M tokens | Query handling | Licence |
|---|---|---|---|---|---|
| OpenAI text-embedding-3-small | 1536, truncatable | 8,192 | 0.02 | none | API |
| OpenAI text-embedding-3-large | 3072, truncatable | 8,192 | 0.13 | none | API |
| Voyage voyage-4 family | 1024 (256–2048) | 32K | 0.02–0.12 | `input_type` required | API |
| Cohere embed-v4 / v5 | 1536–2048 | 128K | about 0.08–0.12 | not captured | API |
| Google gemini-embedding-2 | 3072 (128–3072) | 8,192 | 0.20 | task written into the prompt | API |
| Qwen3-Embedding 0.6B / 4B / 8B | 1024 / 2560 / 4096 | 32K | local | instruction on the query only | Apache-2.0 |
| EmbeddingGemma-300m | 768 (down to 128) | 2,048 | local | prompts required | Gemma terms |
| BGE-M3 | 1024 | 8,192 | local | none | MIT |
| nomic-embed-text-v1.5 | 768 (down to 64) | 8,192 | local | `search_query:` / `search_document:` prefixes | Apache-2.0 |

- OpenAI has released no embedding model newer than text-embedding-3.
- Anthropic offers no embedding model of its own and points to Voyage AI.

**Default.** `text-embedding-3-small` as the cheap starting point. For local use on a Mac: Qwen3-Embedding-0.6B, EmbeddingGemma or BGE-M3 through sentence-transformers.

**Evidence.**

- [I] One independent comparison (September 2026, 200 hard queries) put voyage-4-large, Cohere Embed 5 and gemini-embedding-2 within noise of each other at nDCG@10 of 0.83–0.85, with the best open-weight model at 0.74.
- [V] Voyage reports voyage-4-large beating OpenAI's large model by 14%.
- The maintainers of the public embedding leaderboard wrote (October 2025) that overlap between public test sets and training data inflates scores, and added private held-out sets (RTEB).

**For this project.** Stage 4 ranks models on our own questions. Truncatable dimensions and query prefixes are each tested as a single-variable change.

**Not confirmed:** current leaderboard rankings (not read from the live board); Cohere and Gemini prices came from secondary sources.

Sources:
- https://platform.claude.com/docs/en/build-with-claude/embeddings
- https://developers.openai.com/api/docs/guides/embeddings
- https://blog.voyageai.com/2026/01/15/voyage-4/
- https://aimultiple.com/embedding-models (September 2026)
- https://huggingface.co/blog/rteb (October 2025)
- https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
- https://huggingface.co/google/embeddinggemma-300m
- https://huggingface.co/BAAI/bge-m3

## 4. Vector stores and indexing

**Search methods.** Exact search compares the query with every vector. Approximate methods trade some recall for speed: HNSW (a layered neighbour graph), IVF (cluster, then search a few clusters), and quantization (store each vector in fewer bits).

| Store | Index | Built-in keyword or hybrid search | Filtering | Notes |
|---|---|---|---|---|
| NumPy | exact | no | manual | Also the ground truth for measuring approximate recall. |
| FAISS | Flat, HNSW, IVF, PQ | no | limited | Has Python 3.14 macOS wheels. |
| hnswlib | HNSW | no | not verified | Source-only on Python 3.14; needs a compiler. |
| Chroma | HNSW | full-text locally | yes | Simplest API. |
| LanceDB | IVF-PQ, HNSW, on disk | yes, with rank fusion | SQL, before search by default | Embedded. |
| Qdrant local mode | not documented | on the server | payload filters | Meant for prototyping. |
| sqlite-vec | exact | no | columns | Pre-1.0. |
| pgvector | HNSW, IVFFlat | via Postgres full-text | SQL | Needs Postgres. |
| Milvus Lite | Flat, HNSW, IVF | yes | yes | Embedded. |

**Default.** Exact search until the corpus is large, then HNSW. An embedded store for anything that needs persistence, filtering and hybrid search together.

**Evidence.**

- [I] On 100,000 vectors of 96 dimensions, exact NumPy search took 1.4 ms per query. HNSW recall@10 rose from 0.78 to 0.996 as the search-width parameter went from 10 to 64, at 0.03–0.07 ms.
- Secondary sources put the point where approximate search starts to pay off near 100,000 vectors at 1,536 dimensions. FAISS's own guide recommends exact search for small collections.
- Filtering after search can return fewer than k results. LanceDB filters before search by default; pgvector 0.8 added iterative scans that keep going until enough rows pass.
- Scalar (int8) quantization saves 4x storage and binary saves 32x.

**For this project.** The corpus is a few thousand chunks, so approximate search is unnecessary. Stage 11 builds a larger vector set to show the trade-off, and uses FAISS's HNSW because hnswlib has no Python 3.14 wheel.

**Not confirmed:** whether Chroma's and Qdrant's hybrid search work in local mode.

Sources:
- https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index
- https://opensearch.org/blog/a-practical-guide-to-selecting-hnsw-hyperparameters/ (June 2025)
- https://aicodeinvest.com/approximate-nearest-neighbor-hnsw-recall-latency-benchmark/ (September 2026)
- https://docs.lancedb.com/search/hybrid-search
- https://github.com/pgvector/pgvector

## 5. Retrieval and fusion

**Options.** Dense (one vector per chunk); sparse keyword scoring (BM25) or learned sparse (SPLADE); hybrid; late interaction (one vector per token, as in ColBERT).

**Default.** Hybrid BM25 plus dense, fused with reciprocal rank fusion, then reranked.

**Fusion.**

- Reciprocal rank fusion scores a document as the sum over retrievers of `1 / (k + rank)`, with `k = 60` by convention. It uses ranks only, so it needs no score normalisation and no tuning.
- Weighted fusion computes `α · dense + (1 − α) · sparse` after normalising each score list.
- [I/V] Bruch et al. (2022–23) found weighted fusion beats rank fusion once α is tuned on a few labelled examples, and that rank fusion is sensitive to `k`. No 2025–26 independent re-test was found.

**Other evidence.**

- `bm25s` is the current fast Python BM25 library; `rank_bm25` is adequate on a small corpus.
- [V] Elastic reports rank fusion adding 18% nDCG@10 over BM25 alone.
- Late interaction became easier to use (PyLate, 2025) and faster (MUVERA), and remains an upgrade path, not the default.

**For this project.** Stage 5 builds BM25 and both fusion methods by hand and measures them against each other.

Sources:
- https://arxiv.org/abs/2210.11934 (2022–23)
- https://pypi.org/project/bm25s/
- https://www.elastic.co/search-labs/blog/linear-retriever-hybrid-search
- https://arxiv.org/pdf/2508.03555 (August 2025)

## 6. Query transformation

**Options.** Rewriting; multi-query (several paraphrases, results fused); HyDE (embed a hypothetical answer instead of the question); step-back (ask a more general question first); decomposition into sub-questions; routing between indexes.

**Default.** None is on by default. Rewriting follow-up questions into standalone ones is standard in chat. The rest are added when an evaluation shows a recall problem.

**Evidence.**

- [I] Medrano et al. (March 2026, a production deployment): multi-query raised raw recall, and the gain disappeared after reranking and truncation. Hit@10 fell from 0.51 to 0.48 in several configurations.
- [I] Abe et al. (SIGIR 2025): LLM query expansion hurts when the model lacks knowledge of the topic or the query is ambiguous.
- [I] Weller et al. (EACL 2024): expansion helps weak retrievers and tends to hurt strong ones.
- Each transformation costs one extra LLM call per question.
- No clean independent head-to-head was found for decomposition on multi-hop questions.

**For this project.** Stage 7 comes after reranking and measures each transformation with and without the reranker.

Sources:
- https://arxiv.org/abs/2603.02153 (March 2026)
- https://arxiv.org/html/2505.12694v1 (May 2025)
- https://ar5iv.labs.arxiv.org/html/2309.08541 (2023–24)

## 7. Reranking

**Options.**

- Local cross-encoders, which read the query and the passage together: ms-marco-MiniLM (small), bge-reranker-v2-m3, mxbai-rerank-v2, jina-reranker-v3, Qwen3-Reranker (0.6B to 8B).
- Hosted rerankers: Cohere Rerank 4, Voyage rerank.
- An LLM asked to order a list of passages.

**Default.** Retrieve 50–150 candidates, rerank, keep 5–20.

**Evidence.**

- [I, with a vendor's involvement] One leaderboard (February 2026) ranks hosted rerankers at 265–615 ms per query and an 8B open model at 4.7 seconds.
- [V] Voyage claims its reranker is far cheaper and faster than using a frontier LLM to rerank, with better nDCG@10.
- Returns flatten at about 50–100 candidates according to practitioner write-ups; no controlled study was found.
- No measured reranker latency on Apple Silicon was found.

**For this project.** Stage 6 measures quality and latency on the M4 for three local cross-encoders and an LLM reranker.

Sources:
- https://agentset.ai/rerankers (February 2026)
- https://docs.cohere.com/docs/rerank
- https://docs.voyageai.com/docs/reranker
- https://huggingface.co/Qwen/Qwen3-Reranker-0.6B

## 8. Contextual retrieval

An LLM writes 50–100 tokens placing each chunk in its document. That text is prepended before embedding and before BM25 indexing.

- [V] Anthropic (September 2024) reported the share of questions with missing evidence in the top 20 falling from 5.7% to 3.7% with contextual embeddings, 2.9% adding contextual BM25, and 1.9% adding a reranker, at about $1 per million document tokens with prompt caching.
- [I] Merola and Singh (2025) found a marginal gain over late chunking at much higher compute cost.
- [I] A 2026 study found expensive chunk-enrichment methods rarely give consistent gains over simple chunking.
- No independent replication of Anthropic's figures was found.

**For this project.** Advanced stage A1 measures it against a control that prepends the title and date with no LLM call.

Sources:
- https://www.anthropic.com/news/contextual-retrieval (September 2024)
- https://arxiv.org/abs/2504.19754 (April 2025)
- https://arxiv.org/abs/2608.16586 (August 2026)

## 9. Context assembly

**Default.** 5–20 reranked chunks, deduplicated, each tagged with a source id, placed before the question.

**Evidence.**

- [I] Cuconasu et al. (EMNLP 2025): position effects are marginal in realistic RAG. More than 60% of queries have a highly distracting passage in the top 10, and reordering does no better than shuffling.
- [I] Levy et al. (2025): at a fixed total length, more documents hurt, and near-miss passages hurt more than random ones.
- [V/I] Chroma's "context rot" report (July 2025, 18 models): performance falls as input grows, even on simple tasks.
- [I] LaRA (ICML 2025): strong models do well with long context, weak models do better with RAG, and RAG wins on hallucination and refusal.

**For this project.** Stage 8 measures answer quality against the number of chunks sent, and treats ordering as a minor variable.

Sources:
- https://arxiv.org/html/2505.15561v2 (2025)
- https://arxiv.org/abs/2503.04388 (March 2025)
- https://www.trychroma.com/research/context-rot (July 2025)
- https://arxiv.org/html/2502.09017v1 (February 2025)

## 10. Citations and abstention

**Options.** Inline markers requested in the prompt; Anthropic's native citations; OpenAI file-search annotations; structured output with quotes that you verify yourself.

- **Anthropic citations.** Generally available. Passing retrieved chunks as search-result blocks returns citations that are guaranteed to point into the supplied text. Not compatible with structured outputs.
- **OpenAI.** File search returns file-level citations without a quoted span. For a hand-built pipeline, the practical route is structured output with a quote field, checked by string match.
- [I] Google's "sufficient context" work (ICLR 2025): judge whether the context is enough to answer, not just relevant, and use that to decide when to abstain.
- Small groundedness classifiers exist (for example LettuceDetect, 79% F1 on RAGTruth); a cheap LLM judge is the common fallback.

**For this project.** Stage 9 compares prompted and native citations and three abstention methods.

Sources:
- https://platform.claude.com/docs/en/build-with-claude/citations
- https://platform.claude.com/docs/en/build-with-claude/search-results
- https://developers.openai.com/api/docs/guides/tools-file-search
- https://arxiv.org/pdf/2411.06037 (ICLR 2025)

## 11. Evaluation

**Retrieval metrics.**

- Hit rate@k: is any gold item in the top k.
- Recall@k: share of gold items in the top k. The main metric for RAG, since the generator can only use what was retrieved.
- Precision@k: share of the top k that is relevant.
- MRR: reciprocal rank of the first relevant item.
- nDCG@k: rank-discounted gain; the only one that uses graded relevance.

**Pitfalls when the chunker changes.**

- Gold labels stored as chunk ids go stale. Store document id plus character offsets.
- Recall at a fixed k is not comparable across chunk sizes, because k chunks of 200 tokens and k chunks of 800 tokens are different context budgets. Compare at a fixed token budget, or use token-level recall, precision and overlap (Chroma's method).

**Answer metrics.**

- Faithfulness: share of the answer's claims supported by the retrieved context.
- Correctness against a reference answer.
- Citation precision and recall (ALCE, EMNLP 2023).
- Abstention, scored separately on answerable and unanswerable questions.

**LLM judges.**

- Prefer one binary pass/fail judge per failure type over a graded scale, and validate it against human labels.
- [I] Lee et al. (2025): correct the measured pass rate using the judge's sensitivity and specificity on a human-labelled set.
- Judges show position, verbosity and same-family preference biases, so generate with one vendor and judge with another.
- Temperature 0 does not make output deterministic; cache judge outputs.

**Statistics.**

- [I] Miller (2024): report standard errors, cluster them when questions share a source (clustered errors can be more than three times the naive ones), and analyse paired per-question differences.
- [I] Bowyer et al. (ICML 2025): normal-approximation intervals understate uncertainty below a few hundred items.
- A rough guide: at a 50% score, an unpaired 95% interval is about ±7 points at 200 questions and ±4.4 at 500. Pairing tightens it.

**Frameworks.**

| Tool | Status | Use here |
|---|---|---|
| ranx | maintained, MIT | Checks hand-written retrieval metrics; has significance tests. |
| RAGAS | 0.4.3 (January 2026); no release since; open question about maintenance | Reference metric definitions, pinned. |
| DeepEval | actively released | Maintained alternative for answer metrics. |
| TruLens, Phoenix, Inspect AI, promptfoo | maintained | Not used. |
| pytrec_eval | last release 2020 | Not used. |

**Synthetic test sets.** [I] A 2025 study found they rank retriever configurations consistently with human benchmarks, and do not rank generators reliably.

Sources:
- https://www.trychroma.com/research/evaluating-chunking
- https://arxiv.org/abs/2305.14627 (ALCE)
- https://arxiv.org/abs/2511.21140 (November 2025)
- https://arxiv.org/abs/2411.00640 (November 2024)
- https://arxiv.org/abs/2503.01747 (2025)
- https://arxiv.org/abs/2508.11758 (August 2025)
- https://github.com/AmenRa/ranx
- https://github.com/vibrantlabsai/ragas/issues/3004 (September 2026)

## 12. Corpus candidates

| Dataset | Corpus | Questions | Gold evidence | Licence | Weakness |
|---|---|---|---|---|---|
| **MultiHop-RAG** (chosen) | 609 news articles | 2,556 | sentence plus article | ODC-BY | Clean text; GPT-4-written questions; short answers |
| QASPER | 1,585 NLP papers | 5,049, human-written | paragraph | CC BY 4.0 | Questions assume a known paper |
| Open RAG Bench | 1,000 arXiv PDFs | 3,045 | document plus section | CC BY-NC 4.0 | LLM-written questions |
| T2-RAGBench | 7,318 financial pages | 23,088 | one context id | CC BY 4.0 | Short documents; numeric answers |
| LegalBench-RAG | 714 documents | 6,858 | character span | CC BY 4.0 | No reference answers |
| FinanceBench | 10-K and 10-Q PDFs | 150 | page plus text | CC BY-NC 4.0 | Small; very long PDFs |
| HotpotQA | 10 paragraphs per question | 7,405 | sentence | CC BY-SA 4.0 | Saturated and memorised |
| BEIR subsets | 3–58 thousand documents | 300–650 | document | varies | Retrieval only, no answers |

Why MultiHop-RAG: it is cheap to embed, has reference answers and evidence that maps to character spans, and includes multi-hop, temporal and unanswerable questions, so hybrid search, reranking, decomposition, metadata filtering and abstention all have something to measure. Its weaknesses, measured on our copy, are in [stages/00-setup.md](stages/00-setup.md).

Sources:
- https://arxiv.org/abs/2401.15391 (MultiHop-RAG)
- https://huggingface.co/datasets/yixuantt/MultiHopRAG

## 13. Beyond the core

**GraphRAG.** [I] GraphRAG-Bench (2025–26): plain RAG with a reranker scored 60.9% on fact retrieval against 49.3% for Microsoft GraphRAG; on complex reasoning and summarisation the graph methods led by 8–13 points. Prompt size per query ranged from about 1,000 tokens (plain RAG, HippoRAG 2) to about 331,000 (Microsoft GraphRAG global search). Awareness only in this project.

**Agentic RAG.** Retrieval exposed as a tool with a step cap is the mainstream shape. [I] One 2025 study found agentic keyword search reaching over 90% of vector RAG's performance with no vector index. Built in advanced stage A2.

**Multimodal.** [I] ViDoRe V3 (January 2026): visual page retrievers beat text pipelines (best visual 59.8–66.0 nDCG@10 against 51.0–56.7 for text), text rerankers add about 13 points, and page images as generator context beat extracted text slightly. [I] UniDoc-Bench (2025): combining separate text and image retrieval beats both single-modality retrieval and joint multimodal embeddings. Built in advanced stage A4.

**Caching.** Prompt caching on both vendors makes a repeated prefix cost about a tenth of normal input. Semantic caching (reusing answers for similar questions) is sensitive to its similarity threshold. Prompt caching is used in A1.

**Guardrails.** [I] PoisonedRAG (USENIX Security 2025): five injected texts per target question gave about 90% attack success in a corpus of millions. Built in advanced stage A3.

Sources:
- https://arxiv.org/html/2506.05690 (GraphRAG-Bench)
- https://arxiv.org/abs/2602.23368 (December 2025)
- https://arxiv.org/abs/2601.08620 (ViDoRe V3)
- https://arxiv.org/abs/2510.03663 (UniDoc-Bench)
- https://platform.claude.com/docs/en/build-with-claude/prompt-caching
- https://usenix.org/system/files/usenixsecurity25-zou-poisonedrag.pdf
