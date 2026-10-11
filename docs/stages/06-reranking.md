# Stage 6: Reranking

One variable changes in this stage: a second step after retrieval that reads the question together with each candidate chunk and reorders them. The chunks (recursive, 128 tokens), the embedder (`text-embedding-3-small`), the retriever (hybrid, reciprocal rank fusion at 60) and the prompt stay as they were.

## Concept

**Two ways for a model to compare a question with a chunk.**

| | Bi-encoder (Stages 1 to 5) | Cross-encoder (this stage) |
|---|---|---|
| What the model reads | The question alone, and each chunk alone | The question and one chunk as a single input |
| What is compared | Two vectors, with one dot product | Nothing afterwards: the model's output is the score |
| Work done before the question arrives | Every chunk's vector | None |
| Cost per question | One embedding call and one matrix product | One forward pass per candidate |

In a cross-encoder every word of the question can attend to every word of the chunk, which is why it judges relevance better. Nothing about a chunk can be computed in advance, which is why it cannot be run over 15,149 chunks per question.

**Two-stage retrieval.** A cheap first stage picks candidates from the whole index; the expensive model reorders only those. The reranker adds no chunks: whatever the first stage missed stays missed.

**The text is looked up, not recovered.** A vector cannot be turned back into text. The store keeps each chunk's text at the same row as its vector; search returns row numbers, and the reranker is handed the text at those rows. It ignores the stored vectors and the first-stage scores.

**Candidate depth** is how many first-stage candidates the reranker reads. Deeper gives it more evidence to find and more distractors to rank above that evidence, at a cost in time that grows in proportion.

**Three kinds of reranker.**

| Kind | How it scores | Example here |
|---|---|---|
| Classifier cross-encoder | An encoder with a one-number head, trained on (query, passage, label) triples. One score per pair. | ms-marco-MiniLM, bge-reranker-v2-m3 |
| Language model scored on yes against no | A prompt asks whether the passage answers the query; the score is the logit of "yes" minus the logit of "no" for the next token. One score per pair. | Qwen3-Reranker |
| Listwise LLM | One prompt holds the question and every candidate; the model replies with an ordering. | `gpt-6-luna` |

The first two score each chunk alone and sort. Only the listwise reranker sees the candidates next to each other.

## Options

| Decision | Options | Trade-off |
|---|---|---|
| Reranker | none; a small cross-encoder; a large one; an LLM | Accuracy against latency, and for an LLM a bill per question. |
| Candidate depth | 20, 50, 100 | More evidence within reach against more time and more distractors. |
| What the reranker reads | chunk text; text under a title, source and date line | The header gives the model the source and date a question names. It is a second variable. |
| What comes out | the reranked chunks only; reranked chunks with the rest underneath | Keeping the rest means a shallow depth cannot lose evidence the retriever had. |
| Listwise prompt | every candidate in one prompt; sliding windows of 20 | One prompt is simpler and long; windows cost more calls. Not built here. |
| What the question is | the whole question; its parts | Stage 7's subject. Finding 2 shows why it matters. |

## What we chose

**A reranker slot.** `PipelineConfig` has a `reranker` field that defaults to none, so every earlier config loads and runs unchanged. A reranker reorders chunks the retriever returned, so it is not part of the index fingerprint: every Stage 6 config points at `data/index/chunk-recursive-128`. Code: [rerank/](../../src/ragbasics/rerank/).

**Depth.** The reranker reorders the retriever's top `depth` candidates, and the candidates below that keep their order underneath. The evaluator scores 50 chunks per question, so:

- depth 20 and depth 50 only reorder chunks that were already among the 50, and recall@50 cannot change;
- depth 100 can bring chunks the retriever ranked 51 to 100 into the 50, so recall@50 can change.

**The trace.** `Trace.retrieved` is the retriever's list before reranking and `Trace.candidates` the list after, with the reranker's scores. `Trace.rerank` records the depth, whether the scores came from the cache, pairs truncated and repairs to an LLM's reply. `rag ask` and the app show "was 17" next to a reranked chunk. A reranked evaluation run writes both lists and the rank of each gold sentence before and after.

**What is written by hand.** The weights are a download and the forward pass is the transformers library's. Written in [rerank/cross_encoder.py](../../src/ragbasics/rerank/cross_encoder.py) and [rerank/llm_listwise.py](../../src/ragbasics/rerank/llm_listwise.py):

- pairing each candidate with the question, and what text the model is shown;
- truncation: pairs longer than the model reads are counted, since the model cuts them without an error;
- batching, with pairs sorted by length so a batch is padded to its own longest pair;
- turning the output into a score for both kinds of local model;
- sorting, with equal scores left in first-stage order;
- for the LLM: the prompt, parsing the reply, and repairing a reply that repeats a number, leaves one out or uses one out of range.

**Score cache.** [rerank/cache.py](../../src/ragbasics/rerank/cache.py), a SQLite file keyed by model, revision, input limit, prompt template, question and chunk text. A score depends on the pair alone, so one pass at depth 100 also serves depths 20 and 50, and a repeated run is free and identical. It holds about 340,000 scores in 56 MB. The LLM reranker uses the existing LLM cache.

**Models.**

| Model | Released | Size | Reads up to | Licence | Run |
|---|---|---|---|---|---|
| `cross-encoder/ms-marco-MiniLM-L6-v2` | 2021 | 23M parameters, 91 MB | 512 tokens | Apache-2.0 | yes |
| `Qwen/Qwen3-Reranker-0.6B` | 2025 | 0.6B parameters, 1.2 GB | 32,768 tokens | Apache-2.0 | yes |
| `BAAI/bge-reranker-v2-m3` | 2024 | 568M parameters, 2.3 GB | 8,192 tokens | Apache-2.0 | no: download size |
| `gpt-6-luna`, listwise | | hosted API | | | depths 20 and 50 |

- All three local models load on Python 3.14 with torch 2.14 and transformers 5.19 and use the M4's GPU. Revisions are pinned in each config.
- **Truncation did not bite.** A question is about 54 tokens and a chunk about 92; no pair in any run exceeded the 512-token limit set in the configs.
- **Qwen3-Reranker is loaded at 32-bit precision.** Its own config asks for 16-bit weights, which rounded the two logits to steps of about 0.1 and left many chunks tied.
- **Only the last position goes through Qwen3's output layer.** Asking the model for its full output computes a score for every word of the vocabulary at every position. Skipping that is about 25% faster with identical scores.
- **The LLM is `gpt-6-luna`**, the cheapest tier. `claude-sonnet-5-5` would cost about $7 for depth 50 alone, and Haiku 4.5 is not pinned because of its retirement date. `gpt-6-luna` is also the judge, which matters only for an answer-quality run made behind this reranker.
- **A refused prompt keeps the retriever's order.** OpenAI's usage-policy filter rejected a few prompts (the passages are news, some about crime). Those questions are left unreranked and counted; any other error stops the run.

**How the reranker and depth were chosen.** As in Stage 5, on `questions_pool.jsonl` (284 questions, at most 25 per evidence cluster) by recall within 2,000 tokens, with [scripts/stage06_tune.py](../../scripts/stage06_tune.py), and reported once on the development questions. The pool shares articles and gold sentences with development, so this guards against fitting the wording of the reported questions and not their evidence. The LLM reranker was not tuned on the pool, since every question and depth is a paid call; it is a reported row.

**Configs:** [configs/stage06_rerank/](../../configs/stage06_rerank/), one file per row.

## Measured result

No predictions were written down for this stage.

441 development questions with gold evidence. Differences are paired, in percentage points, with a 95% interval from redrawing evidence clusters. "Best so far" is `retr-hybrid-rrf`. **Every row ranks the same 15,149 chunks, so recall@k, MRR and nDCG@10 are fair between rows**, alongside recall and precision within 2,000 tokens. Regenerate with `python scripts/stage06_sweep.py`; the numbers are in [results/stage06_rerank.csv](../../results/stage06_rerank.csv).

| Reranker | Depth | Recall in 2,000 tok | vs best so far | vs baseline | nDCG@10 | Full support@10 | vs best so far | MRR@10 | Recall@10 | Recall@50 |
|---|---|---|---|---|---|---|---|---|---|---|
| none (best so far) | | 72.9 | | +28.1 (+23.5 to +32.7) | 48.6 | 34.0 | | 57.2 | 59.1 | 84.5 |
| MiniLM | 20 | 72.2 | −0.7 (−1.6 to −0.2) | +27.4 (+22.5 to +31.9) | 43.0 | 21.1 | −12.9 (−18.6 to −8.5) | 57.4 | 50.3 | 84.5 |
| MiniLM | 50 | 60.0 | −12.9 (−15.8 to −10.4) | +15.2 (+10.0 to +19.7) | 41.3 | 15.4 | −18.6 (−25.5 to −13.7) | 57.5 | 46.7 | 84.5 |
| MiniLM | 100 | 57.6 | −15.3 (−18.6 to −12.5) | +12.8 (+7.3 to +17.6) | 41.0 | 14.3 | −19.7 (−26.9 to −14.8) | 57.5 | 46.2 | 75.6 |
| Qwen3 | 20 | 73.0 | +0.1 (0.0 to +0.2) | +28.2 (+23.5 to +32.8) | 58.2 | 44.2 | +10.2 (+6.5 to +15.6) | 67.0 | 67.3 | 84.5 |
| Qwen3 | 50 | 79.7 | +6.7 (+4.2 to +8.9) | +34.9 (+30.3 to +38.6) | 60.0 | 50.6 | +16.6 (+11.5 to +22.9) | 67.3 | 71.2 | 84.5 |
| Qwen3 | 100 | 82.3 | +9.4 (+6.7 to +11.7) | +37.5 (+32.7 to +41.3) | 60.7 | 51.7 | +17.7 (+12.2 to +23.9) | 67.2 | 72.7 | 90.2 |
| `gpt-6-luna` | 20 | 73.0 | +0.1 (0.0 to +0.2) | +28.2 (+23.5 to +32.8) | 67.0 | 46.5 | +12.5 (+8.8 to +17.8) | 78.6 | 70.0 | 84.5 |
| `gpt-6-luna` | 50 | 80.4 | +7.5 (+4.6 to +9.8) | +35.6 (+30.4 to +39.8) | 71.5 | 56.9 | +22.9 (+18.4 to +27.6) | 81.1 | 76.0 | 84.5 |

Recall in 2,000 tokens by question type, with the paired difference from the best so far:

| Reranker | Depth | Comparison (211) | Inference (105) | Temporal (125) |
|---|---|---|---|---|
| none | | 82.0 | 66.0 | 63.5 |
| MiniLM | 50 | 67.5, −14.5 (−18.5 to −10.4) | 49.3, −16.7 (−21.9 to −14.0) | 56.3, −7.2 (−13.3 to −1.3) |
| Qwen3 | 50 | 87.4, +5.4 (+1.8 to +8.4) | 74.3, +8.3 (+3.8 to +13.6) | 71.2, +7.7 (+4.0 to +11.3) |
| Qwen3 | 100 | 90.6, +8.6 (+4.4 to +12.7) | 76.0, +10.0 (+6.2 to +15.2) | 73.6, +10.1 (+5.7 to +14.1) |
| `gpt-6-luna` | 50 | 89.0, +7.0 (+4.3 to +9.6) | 76.6, +10.6 (+6.1 to +15.2) | 69.1, +5.6 (+0.7 to +10.4) |

### Finding 1: a small cross-encoder made retrieval worse

MiniLM at depth 50 loses 12.9 points of recall in 2,000 tokens (−15.8 to −10.4) and more than halves the share of questions with every gold sentence in the top 10, from 34.0 to 15.4. It gets worse with depth, and the pool questions show the same pattern.

- **It is not a bug.** The hand-written scoring matches the library's to the last digit. Among each question's top 100 candidates, MiniLM ranks a gold chunk above a non-gold one 77.9% of the time, where 50% would be chance. The hybrid order it replaces does so 87.2% of the time.
- **MRR does not move** (57.2 to 57.5) while full support collapses. It still puts one relevant chunk near the top and pushes the question's other evidence down.
- So "add a reranker" is not a gain by default. A reranker replaces the first stage's order, and here that order was already better than this model's.

### Finding 2: MiniLM fails because it scores each chunk against the whole question

These questions ask about several facts at once, and each gold chunk supports one of them. MiniLM was trained on short web queries where a relevant passage answers the whole query, so a chunk that covers one fact in three looks like a poor match. A chunk that mentions several of the question's names without stating any needed fact can outrank it; for one question the top six after reranking were the same line of site boilerplate from Engadget.

The test, in [scripts/stage06_clauses.py](../../scripts/stage06_clauses.py): cut each question into clauses with a fixed rule and no model (at commas and at "and", "while", "but", "before", "after", with pieces under five words glued back; a median of 3 clauses), and score every candidate against each clause as well as against the whole question.

Share of (gold chunk, non-gold chunk) pairs ordered correctly, among each question's top 100 candidates:

| Ordering | All | Comparison | Inference | Temporal |
|---|---|---|---|---|
| hybrid, no reranker | 87.2 | 90.9 | 85.4 | 82.3 |
| MiniLM, whole question | 77.9 | 80.8 | 70.9 | 79.0 |
| MiniLM, best clause | 88.5 | 91.4 | 88.2 | 83.8 |
| MiniLM, round robin over clauses | 89.7 | 92.7 | 89.6 | 84.6 |

- **Moving from the whole question to its best clause raises a gold chunk's score by 2.58 at the median** (971 chunks) and leaves the other candidates where they were, at −0.12 (43,129 chunks).
- "Best clause" takes a chunk's highest score over the clauses. "Round robin" lets each clause rank the candidates and takes every clause's first choice, then every clause's second, and so on.

What that does to the metrics, same model and same candidates:

| Ordering | Depth | Recall in 2,000 tok | vs hybrid | nDCG@10 | Full support@10 |
|---|---|---|---|---|---|
| whole question | 50 | 60.0 | −12.9 (−15.8 to −10.4) | 41.3 | 15.4 |
| best clause | 50 | 74.2 | +1.2 (−1.5 to +3.7) | 55.5 | 42.2 |
| round robin | 50 | 76.0 | +3.0 (+0.5 to +5.2) | 53.8 | 43.5 |
| round robin | 100 | 76.8 | +3.9 (+1.2 to +6.3) | 53.9 | 43.8 |

Splitting the question turns a 13-point loss into a 3-point gain: +16.0 (+13.2 to +18.9) against whole-question scoring at depth 50.

Limits of this test:

- A clause is both the right part of the question and a shorter query, closer to what MiniLM was trained on. The test shows that splitting helps and does not separate the two reasons.
- It is a diagnostic on the development questions with a crude splitter. Nothing was chosen from it and these are not Stage 6 rows: splitting the question is a second variable, and it is Stage 7's subject.
- It was run on MiniLM only. Whether Qwen3 also gains from split questions is deferred to Stage 7.

### Finding 3: Qwen3-Reranker adds 9 points, on every question type

At depth 100, 82.3% of the gold evidence is within 2,000 tokens against 72.9%: +9.4 (+6.7 to +11.7), and +37.5 against the Stage 1 baseline. The share of questions with every gold sentence in the budget goes from 49.9% to 64.9%, and in the top 10 from 34.0% to 51.7%.

- **Every type gains about 9 to 10 points**, including temporal questions (63.5 to 73.6), the weak type after Stage 5.
- Same input as MiniLM, the whole question and one chunk, and the opposite result. The two differ in size (0.6B parameters against 23M), in training data, and in form: Qwen3 is a language model reading an instruction. This stage does not separate which of those matters.
- It is the second-largest effect on retrieval in the project so far, level with hybrid retrieval (+9.1) and behind chunk size (+19.0).

### Finding 4: depth helps a good reranker and hurts a poor one

| Qwen3 | Recall in 2,000 tok | nDCG@10 | Full support@10 | Recall@50 |
|---|---|---|---|---|
| depth 50 against 20 | +6.7 (+4.2 to +8.8) | +1.9 (+0.8 to +2.9) | +6.3 (+3.4 to +9.6) | 0 |
| depth 100 against 50 | +2.6 (+1.6 to +3.7) | +0.6 (+0.1 to +1.1) | +1.1 (−1.0 to +3.2) | +5.7 (+4.0 to +6.9) |

- **Depth 20 cannot move recall in 2,000 tokens.** The budget holds about 21 chunks of 92 tokens, so reordering the top 20 changes their order and not which of them are inside. It still raises nDCG@10 by 9.6 points. Whether depth 20 is "enough" depends on how many chunks the prompt takes.
- **Depth 100 raises recall@50 from 84.5 to 90.2.** That is above the 87.7% that the union of the dense and BM25 top-50 lists held in Stage 5, because the reranker reads to rank 100 of the fused list.
- **The gain per added candidate falls:** 6.7 points for 30 more candidates, then 2.6 for 50 more.
- For MiniLM every step deeper is worse (72.2, 60.0, 57.6): more candidates are more chances to put a distractor first.

### Finding 5: the LLM matches Qwen3 on evidence found and is far better at the top of the list

| `gpt-6-luna` against Qwen3 | Recall in 2,000 tok | nDCG@10 | Full support@10 |
|---|---|---|---|
| depth 20 | 0.0 | +8.8 (+5.4 to +12.1) | +2.3 (+0.5 to +4.1) |
| depth 50 | +0.7 (−0.8 to +2.2) | +11.5 (+7.1 to +15.5) | +6.3 (+1.4 to +10.7) |

- **Within 2,000 tokens the two are level.** With about twenty chunks in the prompt, either delivers the same evidence.
- **In the first ten the LLM is well ahead**: nDCG@10 71.5 against 60.0, MRR@10 81.1 against 67.3. It sees the candidates side by side, which a pair scorer cannot.
- **Its replies need repair, more so with more passages.**

| Depth | Refused by the provider | Repaired | Cut off at the output limit | Billed |
|---|---|---|---|---|
| 20 | 7 of 500 | 38 | 0 | $0.30 |
| 50 | 2 of 500 | 107 | 3 | $0.54 |

- **Depth 100 was not run.** It was estimated at about $0.90, and at depth 50 one reply in five already needed repair and three ran out of output tokens.
- **Output tokens were five times the first estimate.** The reply is about 100 tokens; the model spends about 700 more reasoning first, billed as output. The estimator in `rag eval` now uses the measured figure.

### Finding 6: reranking recovers most of what fusion lost

Stage 5 found 57 questions where the fused list left less evidence within 2,000 tokens than the better of dense and BM25 had alone (81.3% for that retriever).

| Reranker | Depth | Recall in 2,000 tok on these 57 | Back at or above the better retriever |
|---|---|---|---|
| none | | 36.0 | 0 |
| MiniLM | 50 | 44.7 | 12 |
| Qwen3 | 50 | 65.6 | 31 |
| Qwen3 | 100 | 71.9 | 37 |
| `gpt-6-luna` | 50 | 67.5 | 34 |

Qwen3 at depth 100 brings 37 of the 57 back and most of the lost evidence, though not all of it: 71.9 against 81.3. The repair works because the reranker reads each candidate and does not need the two retrievers to agree on it.

### Finding 7: quality against latency

Measured on the M4's GPU with the score cache off, the model warm and a batch size of 32. Per question: the median and the 95th percentile. Qwen3's figures are from 20, 20 and 8 questions and are rough. A second run of the same measurement came out 5 to 13% slower on every row, so read these to one significant figure. Regenerate with `python scripts/stage06_sweep.py --timing`.

| Reranker | Depth | Recall in 2,000 tok | Full support@10 | Median | 95th percentile |
|---|---|---|---|---|---|
| none | | 72.9 | 34.0 | | |
| MiniLM | 20 | 72.2 | 21.1 | 45 ms | 53 ms |
| MiniLM | 50 | 60.0 | 15.4 | 106 ms | 122 ms |
| MiniLM | 100 | 57.6 | 14.3 | 200 ms | 238 ms |
| Qwen3 | 20 | 73.0 | 44.2 | 2.4 s | 2.9 s |
| Qwen3 | 50 | 79.7 | 50.6 | 5.9 s | 7.0 s |
| Qwen3 | 100 | 82.3 | 51.7 | 12.0 s | 14.3 s |

- **Qwen3 is about 55 times slower than MiniLM per pair**: about 8 to 9 pairs per second against about 470. Loading takes 0.5 s for MiniLM and 1 s for Qwen3 once the weights are in the disk cache.
- Hybrid retrieval before it takes a few milliseconds plus 50 to 200 ms for the question's embedding. The reranker is now almost all of the retrieval time.
- `gpt-6-luna` was not timed. Its runs took several seconds per call with six calls in flight, and that includes the network.
- These are laptop numbers. A server GPU would run Qwen3 an order of magnitude faster; the ratio between the two models is the part that carries over.

### Finding 8: three open candidates from earlier stages, tested behind the reranker

Each row is Qwen3 at depth 100 with one more thing changed.

| Change | Recall in 2,000 tok | nDCG@10 | Full support@10 |
|---|---|---|---|
| weighted fusion in place of RRF | +0.5 (0.0 to +0.9) | +0.4 (0.0 to +0.9) | +0.2 (−0.8 to +1.3) |
| Qwen3 embeddings in place of `text-embedding-3-small` | −0.3 (−1.4 to +1.0) | +0.4 (−0.4 to +1.1) | 0.0 (−1.7 to +1.5) |
| the reranker also reads title, source and date | +1.0 (−0.9 to +2.8) | −2.6 (−5.6 to +0.1) | −5.2 (−10.7 to −0.7) |

- **Weighted fusion is closed.** Its Stage 5 advantage was the order of the first ten chunks (+4.4 nDCG@10). The reranker rewrites that order, and +0.4 is left. Reciprocal rank fusion stays.
- **Qwen3 as the embedder is closed for any config with this reranker.** Its Stage 4 advantage was also order near the top (+4.7 nDCG@10); behind the reranker nothing is left, and its candidates hold slightly less evidence at depth 100 (90.3% against 91.3%). It would matter again only in a pipeline with no reranker.
- **The header in front of the reranker is not taken.** Recall in 2,000 tokens is level and temporal questions gain 3 points (73.6 to 76.5), but full support@10 falls 5.2. I have no measured reason for the loss at the top. The header stays an open candidate for the retrieval indexes (A1, or Stage 8), where Stage 5 measured it helping.

### Did choosing on the pool cost anything?

| | Chosen on the pool | On development | Best on development |
|---|---|---|---|
| Reranker and depth | Qwen3, depth 100 (80.3 on the pool, against 69.2 with none) | 82.3 | 82.3, the same row |

Every ordering agrees between the two question sets: MiniLM worse with depth, Qwen3 better, depth 100 best.

### Framework equivalent

[framework_equivalents/cross_encoder_library.py](../../framework_equivalents/cross_encoder_library.py) uses sentence-transformers' `CrossEncoder`, which pairs, tokenises with truncation, batches, runs the model and sorts.

- `test_classifier_scores_and_order_match_sentence_transformers` checks scores and order for MiniLM on a sample with a limit low enough to force truncation. On three pairs scored at full length the two agreed to every printed digit.
- `test_yes_no_scores_match_the_recipe_on_the_qwen3_model_card` checks the yes-against-no scoring against the code on Qwen3-Reranker's model card.
- Both are skipped without the `local` extra or when the weights are not on the machine; no test downloads a model.

What differs: the library applies whatever activation the model's config names (none for the ms-marco models, a sigmoid for some others), truncates without counting, and has no score cache.

### New best config

**Hybrid retrieval, then Qwen3-Reranker over the top 100:** [configs/stage06_rerank/qwen3-100.yaml](../../configs/stage06_rerank/qwen3-100.yaml), and [qwen3-100-top25.yaml](../../configs/stage06_rerank/qwen3-100-top25.yaml) with 25 chunks in the prompt. `BEST_RUN` in [eval/report.py](../../src/ragbasics/eval/report.py) is now `rerank-qwen3-100`.

- Chosen on the pool by recall in 2,000 tokens, and +9.4 (+6.7 to +11.7) on development against hybrid retrieval alone.
- **Qwen3 over the LLM** because the two are level on the metric this project decides retrieval on, and Qwen3 is free, local, deterministic and never refuses a prompt. The LLM's advantage is the order of the first ten chunks. That makes it the open candidate if Stage 8 settles on a prompt of five to ten chunks.
- **Depth 100 over 50** for 2.6 more points (+1.6 to +3.7). It costs about 12 seconds per question here against 6. This project caches every score and nobody waits on a query, so it takes the quality. For an interactive system on this hardware, depth 50 or 20 is the choice, and the table above is how to make it.
- **No answer-quality run was made.** `rag eval --estimate` puts it at $1.37 for `qwen3-100-top25`.

**Open items carried forward:** human labels for the judge (before Stage 8); a gold-evidence row with source and date headers (with Stage 8); the 64-token chunk candidate (Stage 8); the listwise LLM reranker as the candidate for a short prompt (Stage 8); the title, source and date header in the retrieval indexes (A1, or Stage 8); and whether Qwen3-Reranker gains from split questions as MiniLM did (Stage 7). Closed in this stage: weighted fusion, and Qwen3 as the embedder.

### Cost

Stage 6 spent $0.85, taking the project to $7.63 of $40: two `gpt-6-luna` reranking runs at $0.30 and $0.54, including $0.03 for a first attempt that stopped on a refused prompt. The local models and every embedding were free. The estimate given at the start was about $1 to $2 for three LLM depths; two were run.

## When you would choose differently in production

- **Measure the reranker on your own questions before adding it.** Here one made retrieval 13 points worse and another 9 points better, on the same candidates.
- **Match the reranker to the shape of your queries.** A small cross-encoder trained on short single-fact queries suits short single-fact queries. For long questions that need several pieces of evidence, use a larger model, or split the question first (Stage 7).
- **Set the depth from your latency budget and the size of your prompt.** If the prompt takes 20 chunks, reranking 20 changes only their order. Rerank several times as many candidates as you will use.
- **Choose a small cross-encoder when the budget is tens of milliseconds** and it tests well on your queries: it was 55 times faster here.
- **Choose an LLM reranker when the order of the first few results is what matters** and seconds of latency and a bill per query are acceptable. Plan for replies that need repair, for an output limit, and for the provider refusing some prompts.
- **Use a hosted reranking API** (Cohere, Voyage, Jina) when you want a strong model without running a GPU. None was tested here; the adapter is one class with a `rerank` method.
- **Run rerankers on a GPU server, and batch across requests.** Laptop latency is not what a deployment would see.
- **Cache scores when questions repeat.** A score depends only on the pair, so it survives any change to the retriever.
- **Once a reranker is in place, spend less on the order the first stage produces.** Fusion method and embedder choice stopped mattering here; what the first stage has to deliver is the evidence somewhere in the candidates.

## Questions you should now be able to answer

1. Why is a cross-encoder more accurate than comparing two vectors, and why not run it over the whole corpus?
2. How does the reranker get the text of a chunk when the retriever worked with vectors?
3. How many candidates are enough?
4. Why did reranking the top 20 change nDCG@10 and not recall within 2,000 tokens?
5. Why did MiniLM push down chunks that held the evidence?
6. What does a listwise LLM reranker see that a cross-encoder does not, and what does it cost?
7. Why did weighted fusion and the Qwen3 embedder stop mattering?
8. What does a reranker add in latency on this machine?

<details>
<summary>Short answers</summary>

1. It reads the question and the chunk as one input, so each word of one can attend to each word of the other; two vectors are each computed without sight of the other text. It needs one forward pass per pair and nothing can be precomputed: 15,149 passes per question here.
2. The store keeps each chunk's text at the same row as its vector. Search returns rows; the text is looked up. A vector cannot be turned back into text.
3. It depends on the reranker and the prompt. For Qwen3, 50 was worth 6.7 points over 20 and 100 another 2.6, at twice the time each step. For MiniLM more was worse.
4. Two thousand tokens hold about 21 of these chunks. Reordering the top 20 changes which come first and not which are inside the budget.
5. It scores a chunk against the whole question, and each gold chunk supports one fact of several. Scored against the clause it supports, a gold chunk's score rose by 2.58 at the median while other candidates' did not move, and the same model went from 13 points below the hybrid list to 3 above.
6. The candidates next to each other, so it can prefer the chunk that states a fact over several that mention it. It was 11.5 points better on nDCG@10 than Qwen3 at depth 50, for about half a cent per question, seconds of latency, replies that needed repair one time in five, and a few refused prompts.
7. Both were better only in how they ordered the first ten chunks. The reranker rewrites that order, so what is left of the first stage's contribution is which candidates it hands over, and there they were level.
8. About 0.1 seconds for MiniLM and 6 seconds for Qwen3-Reranker at depth 50, against a few milliseconds for the hybrid search itself.

</details>
