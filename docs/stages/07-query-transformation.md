# Stage 7: Query transformation

One variable changes in this stage: what the retriever (and, in one family of rows, the reranker) is given in place of the question as asked. The chunks (recursive, 128 tokens), the embedder (`text-embedding-3-small`), the retriever (hybrid, reciprocal rank fusion at 60), the Stage 6 reranker (Qwen3-Reranker-0.6B) and the prompt stay as they were. The generator always answers the question as asked.

## Concept

**The question is not always the best search query.** A user writes a question for a person; a retriever compares it with chunks of articles. Query transformation puts a step in between that turns the question into one or more queries closer to what the evidence looks like.

| Transformation | What it produces | Calls |
|---|---|---|
| Rewrite | One search query, usually a keyword string | 1 LLM call |
| Multi-query | Several paraphrases of the question, each searched | 1 LLM call |
| HyDE (hypothetical document embeddings) | A made-up passage that would answer the question, embedded as a passage | 1 LLM call |
| Decomposition | One sub-question per fact the question needs | 1 LLM call |
| Clause splitter | The question cut at commas and conjunctions by a fixed rule | none |

The clause splitter is the no-LLM control: the rule from the Stage 6 diagnostic, which needs no model and costs nothing.

**Several queries give several lists, and they have to be combined.** Two ways were built:

- **Reciprocal rank fusion** (Stage 5): a chunk's score is the sum of 1/(60 + rank) over the lists. A chunk found by several lists rises.
- **Round robin**: take every list's first choice, then every list's second, and so on, skipping chunks already taken. Each list gets an equal share of the top.

They reward different things. Fusion rewards agreement between lists; round robin guarantees each list a place. Sub-questions about different facts *should* disagree, so fusion buries what only one of them finds.

**What gets searched.** A transformation can replace the question ("queries") or be searched alongside it ("both").

**Two places a transformation can act once there is a reranker.**

| | Changes the candidates | Changes what the reranker scores against |
|---|---|---|
| What differs | The retriever searches with the transformed queries; the reranker scores the candidates against the question | The candidates are the question's own; the reranker scores them against each part and the orderings are merged |
| Called here | "in front of the reranker" | "the reranker scores the parts" |
| Answers | Does the transformation find evidence the reranker can use? | Does Qwen3 gain from split questions, as MiniLM did in Stage 6? |

**Follow-up questions in a chat** are a different job: "When did they first beat them?" needs the earlier turn to be searchable at all. A rewriting step turns it into a question that stands alone. It is built here for the app and checked by hand; the evaluation questions are single-turn, so it is not in the results tables.

## Options

| Decision | Options | Trade-off |
|---|---|---|
| Transformation | none; rewrite; multi-query; HyDE; decomposition; a fixed rule | An LLM call per question (latency, cost, refusals) against queries closer to the evidence. |
| What is searched | the transformed queries; the question and the queries | Keeping the question protects against a bad transformation, at one more search. |
| Combining lists | reciprocal rank fusion; round robin | Agreement between lists against a fair share for each. |
| Where it acts with a reranker | the candidates; what the reranker scores against | More evidence among the candidates against a better judgement of each one. |
| Merging the reranker's orderings | round robin; best score per chunk | As above, for the reranker's per-part orderings. |
| LLM | `gpt-6-luna`; a larger model | The cheapest tier that passed a spot check of five questions per transformation. |
| Libraries | hand-written; LangChain `MultiQueryRetriever` and `HypotheticalDocumentEmbedder`; LlamaIndex `SubQuestionQueryEngine` | See "Framework equivalent". |

## What we chose

**A transformation slot.** `PipelineConfig` has a `query_transform` field that defaults to none, so every earlier config loads and runs unchanged, and a `followup` field for the chat. Code: [query/](../../src/ragbasics/query/).

- [query/base.py](../../src/ragbasics/query/base.py): the routing every transformation shares: `retrieve_with` (queries, original or both), `rerank_with` (the same three), `combine` (rrf or round_robin), `rerank_combine` (round_robin or max) and `depth`.
- [query/llm.py](../../src/ragbasics/query/llm.py): the four LLM transformations, their prompts, parsing the reply (stripping numbering, removing repeats, cutting to the limit, all counted), and the fallback.
- [query/clauses.py](../../src/ragbasics/query/clauses.py): the clause splitter, moved from the Stage 6 script unchanged.
- [query/followup.py](../../src/ragbasics/query/followup.py): rewriting a chat message into a question that stands alone, from the last three turns.
- Round robin is in [retrieval/fusion.py](../../src/ragbasics/retrieval/fusion.py), and merging the reranker's per-part orderings is `merge` in [rerank/base.py](../../src/ragbasics/rerank/base.py).

**Depth per query.** With several queries, each list is fetched to the reranker's depth (100) before combining, so combining has something to choose from; the combined list is then cut to the usual length.

**HyDE's passage is embedded as a passage.** The made-up passage is searched with `kind="document"`, so an embedder with separate query and passage prefixes (Qwen3 in Stage 4) treats it as a passage. `text-embedding-3-small` has no prefixes, so here it changes nothing.

**Failure falls back to the question.** A refused prompt, or a reply that parses to nothing, is searched with the question as asked and counted. There were no refusals. HyDE returned an empty passage for 10 of 784 questions (9 of the 500 development questions): `gpt-6-luna` spent the whole 2,000-token output limit reasoning and wrote no passage.

**The trace.** `Trace` keeps the question as asked (and, in a chat, the message as typed), the transformed queries with their role, each query's own list, what the reranker scored against, and the combined list. `rag ask` and the app show the queries and each chunk's rank in each query's list. An evaluation run writes all of it per question.

**Caching.** Every LLM reply is in the existing LLM cache and every query vector in the embedding cache, so a repeated run is free and identical. The tuning sweep scored about 200,000 new reranker pairs; all are in the Stage 6 score cache.

**Model.** `gpt-6-luna`, the cheapest tier, after a spot check of five questions per transformation. Haiku 4.5 is not pinned because of its retirement date. Output was far below the reranker's: a mean of about 140 tokens for rewrite, 250 for multi-query, 410 for HyDE and 270 for decomposition, most of it reasoning.

**How the forms were chosen.** As in Stages 5 and 6, on `questions_pool.jsonl` (284 questions, at most 25 per evidence cluster) by recall within 2,000 tokens, with [scripts/stage07_tune.py](../../scripts/stage07_tune.py), and reported once on the development questions. Each transformation was tried searched alone and with the question, combined both ways, with no reranker. Every one of the five chose **the question and the queries, combined by round robin**. The split rows were chosen the same way, behind Qwen3 at depth 50. The full sweep is in [results/stage07_tuning.csv](../../results/stage07_tuning.csv).

**Configs:** [configs/stage07_query/](../../configs/stage07_query/), one file per row: each transformation with and without the reranker, and two split rows.

### What the transformations wrote

One development question, from the five spot-checked:

> **q_0033 (inference):** What company, recently reported by TechCrunch to have made significant profitability in the third quarter, has also faced criticism for not adequately preventing incidents and streamlining driver sign-ups, despite being aware of safety concerns since 2014 and introducing new app features to address them?

| Transformation | Output |
|---|---|
| Clauses (rule) | "What company, recently reported by TechCrunch to have made significant profitability in the third quarter" / "has also faced criticism for not adequately preventing incidents and streamlining driver sign-ups" / "despite being aware of safety concerns since 2014" / "introducing new app features to address them?" |
| Rewrite | "TechCrunch company significant profitability third quarter criticism not adequately preventing incidents streamlining driver sign-ups aware of safety concerns since 2014 new app features address safety concerns" |
| Multi-query | Three reorderings of the same sentence, each keeping every detail, for example "Which company, after TechCrunch recently reported it made significant profits in the third quarter, has drawn criticism over…" |
| HyDE | "Uber reported significant profitability in the third quarter, according to TechCrunch. The company has faced criticism over its handling of safety incidents and streamlined driver sign-ups, despite awareness of safety concerns dating to 2014 and the introduction of new app features." |
| Decompose | "What company did TechCrunch recently report had made significant profitability in the third quarter?" / "What company has faced criticism for not adequately preventing incidents and streamlining driver sign-ups?" / "What company has been aware of safety concerns since 2014 and introduced new app features to address them?" |

- **Rewrite and multi-query stay close to the question.** The rewrite is the question as keywords; the paraphrases reorder it. Neither adds a word the question lacks.
- **HyDE answers from the model's memory.** It wrote "Uber" here, and "Huawei" for a question whose reference answer is "Insufficient information." On a public news corpus that can help retrieval for the wrong reason: the model already knows the answer. On a private corpus it would not know it, and a confident wrong name is a query for the wrong documents.
- **Decomposition writes questions that stand alone; the rule writes fragments**, such as "and Argentina last summer on October 14, 2023". A development question gets 2.3 sub-questions and 3.1 clauses on average.

## Measured result

No predictions were written down for this stage.

441 development questions with gold evidence. Differences are paired, in percentage points, with a 95% interval from redrawing evidence clusters. Each row is compared with the same pipeline without the transformation: hybrid retrieval (`retr-hybrid-rrf`) with no reranker, Qwen3 at depth 100 (`rerank-qwen3-100`, the best so far) in front of the reranker, and Qwen3 at depth 50 (`rerank-qwen3-50`) for the split rows. **Every row ranks the same 15,149 chunks, so recall@k, MRR and nDCG@10 are fair between rows**, alongside recall within 2,000 tokens. Regenerate with `python scripts/stage07_sweep.py`; the numbers are in [results/stage07_query.csv](../../results/stage07_query.csv).

**No reranker** (against hybrid retrieval: 72.9 recall in 2,000 tokens, 48.6 nDCG@10, 34.0 full support@10)

| Transformation | Queries searched | Recall in 2,000 tok | vs hybrid | nDCG@10 | vs hybrid | Full support@10 | vs hybrid |
|---|---|---|---|---|---|---|---|
| Rewrite | 2.00 | 73.1 | +0.2 (−0.7 to +1.3) | 49.1 | +0.6 (0.0 to +1.0) | 34.5 | +0.5 (−1.0 to +1.5) |
| Multi-query | 4.00 | 73.3 | +0.4 (−0.7 to +1.4) | 49.5 | +1.0 (+0.2 to +1.7) | 37.4 | +3.4 (+1.3 to +6.1) |
| HyDE | 1.98 | 74.5 | +1.6 (−0.2 to +3.4) | 49.8 | +1.2 (+0.1 to +2.5) | 37.6 | +3.6 (+1.9 to +6.0) |
| Decomposition | 3.26 | **78.1** | **+5.2 (+2.8 to +7.3)** | 54.5 | +5.9 (+4.1 to +7.3) | 43.5 | +9.5 (+5.3 to +13.7) |
| Clauses (rule) | 4.11 | 74.5 | +1.5 (−0.9 to +4.3) | 50.8 | +2.2 (+0.5 to +3.7) | 41.3 | +7.3 (+3.3 to +11.5) |

"Queries searched" includes the question.

**In front of the reranker** (against Qwen3 at depth 100: 82.3, 60.7, 51.7)

| Transformation | Recall in 2,000 tok | vs Qwen3 alone | nDCG@10 | vs Qwen3 alone | Full support@10 | vs Qwen3 alone | Evidence among the 100 candidates |
|---|---|---|---|---|---|---|---|
| none | 82.3 | | 60.7 | | 51.7 | | 91.3 |
| Rewrite | 82.7 | +0.4 (−0.2 to +0.9) | 60.9 | +0.2 (−0.1 to +0.7) | 52.4 | +0.7 (−0.3 to +1.8) | 91.4 |
| Multi-query | 82.9 | +0.5 (−0.4 to +1.4) | 61.1 | +0.4 (−0.2 to +1.0) | 52.2 | +0.5 (−0.7 to +1.4) | 91.5 |
| HyDE | 82.9 | +0.6 (−0.4 to +1.4) | 61.2 | +0.6 (−0.1 to +1.2) | 52.4 | +0.7 (−0.7 to +2.3) | 92.8 |
| Decomposition | 82.6 | +0.2 (−1.5 to +1.9) | 61.2 | +0.5 (−0.4 to +1.4) | 52.6 | +0.9 (−0.9 to +3.1) | 92.3 |
| Clauses (rule) | 82.9 | +0.6 (−0.9 to +2.0) | 61.7 | +1.0 (0.0 to +1.9) | 54.6 | +2.9 (+0.7 to +5.0) | 92.0 |

**The reranker scores the parts** (against Qwen3 at depth 50 on the same candidates: 79.7, 60.0, 50.6)

| Parts | Merged by | Recall in 2,000 tok | vs Qwen3 alone | nDCG@10 | vs Qwen3 alone | Full support@10 | vs Qwen3 alone |
|---|---|---|---|---|---|---|---|
| Clauses (rule) | best score | 79.8 | +0.2 (−1.2 to +1.5) | 61.4 | +1.4 (−1.1 to +3.5) | 50.6 | 0.0 (−3.2 to +2.6) |
| Decomposition | round robin | 80.4 | +0.8 (−0.4 to +1.8) | 63.8 | **+3.8 (+2.1 to +5.3)** | 55.3 | **+4.8 (+1.8 to +7.5)** |

Both score against the question and each part.

Recall in 2,000 tokens by question type, with the paired difference from each row's reference:

| Transformation | Comparison (211) | Inference (105) | Temporal (125) |
|---|---|---|---|
| *no reranker* | 82.0 | 66.0 | 63.5 |
| Rewrite | 82.9, +0.9 (−0.9 to +2.6) | 65.0, −1.0 (−3.5 to 0.0) | 63.5, 0.0 (−1.3 to +1.8) |
| Multi-query | 82.9, +0.9 (−0.6 to +2.4) | 65.9, −0.2 (−3.2 to +1.9) | 63.3, −0.1 (−2.1 to +1.7) |
| HyDE | 83.6, +1.7 (−0.5 to +3.5) | 69.0, +2.9 (−0.9 to +8.2) | 63.9, +0.4 (−2.8 to +4.5) |
| Decomposition | 87.1, +5.1 (+1.5 to +8.3) | 73.6, +7.5 (+2.8 to +14.2) | 66.8, +3.3 (+1.0 to +5.9) |
| Clauses (rule) | 82.9, +0.9 (−2.9 to +4.7) | 72.4, +6.3 (+0.6 to +15.8) | 62.0, −1.5 (−4.7 to +1.6) |
| *Qwen3, depth 100* | 90.6 | 76.0 | 73.6 |
| Rewrite | 90.8, +0.2 (−0.6 to +1.1) | 77.1, +1.0 (0.0 to +1.4) | 73.9, +0.3 (0.0 to +0.9) |
| Multi-query | 90.6, 0.0 (−1.1 to +0.8) | 77.7, +1.7 (−0.5 to +3.0) | 74.1, +0.5 (0.0 to +1.4) |
| HyDE | 90.6, 0.0 (−0.7 to +0.6) | 77.9, +1.9 (0.0 to +3.5) | 74.0, +0.4 (−1.3 to +2.5) |
| Decomposition | 90.6, 0.0 (−2.0 to +1.7) | 77.1, +1.0 (−0.5 to +2.5) | 73.6, 0.0 (−2.4 to +3.1) |
| Clauses (rule) | 91.2, +0.6 (−0.9 to +2.0) | 77.4, +1.3 (−3.9 to +4.7) | 73.7, +0.1 (−1.5 to +2.1) |

### Finding 1: without a reranker, decomposition is the one transformation that clearly helps

+5.2 points of recall in 2,000 tokens (+2.8 to +7.3), +5.9 nDCG@10 and +9.5 full support@10, with every question type gaining: inference most (+7.5), then comparison (+5.1) and temporal (+3.3). The share of questions with every gold sentence in the budget goes from 49.9% to 59.4%.

- **It works for the reason Stage 6 found.** These questions ask about two to four facts at once, and each gold chunk supports one. A sub-question about one fact is a query that chunk can match on its own.
- **HyDE is second** (+1.6, interval crossing zero) and helps inference questions most (+2.9), the type of the example above, where the model named the answer.
- **Rewrite and multi-query do nothing to recall** (+0.2 and +0.4). Their queries keep every detail of the question in a new order, so they find what the question found. Multi-query lifts full support@10 by 3.4, the only place having four lists shows.
- **The free clause rule gets a third of decomposition's gain** (+1.5) and most of its gain on inference questions (+6.3 against +7.5), but loses on temporal ones (−1.5), where it cuts a date away from the fact it qualifies.

### Finding 2: how the lists are combined decides whether splitting helps or hurts

On the development questions, searched with the transformed queries alone:

| Transformation | Reciprocal rank fusion | Round robin |
|---|---|---|
| Decomposition | −10.3 (−13.6 to −6.5) | +4.6 (+2.1 to +7.0) |
| Clauses (rule) | −10.2 (−14.3 to −5.6) | +0.4 (−2.2 to +3.4) |
| Multi-query | +0.4 (−0.8 to +2.0) | +0.4 (−0.6 to +1.6) |

The same sub-questions are 15 points apart depending on how their lists are combined.

- **Fusion rewards agreement, and sub-questions should not agree.** A chunk about the third-quarter results is found by one sub-question; a chunk found by every sub-question is one that mentions all the names without stating any of the facts. Fusion puts the second kind on top.
- **Round robin gives each sub-question its share of the top**, which is what a question needing one chunk per fact wants.
- **For paraphrases it makes no difference**, because paraphrases find the same chunks and both methods agree on them.
- **Adding the question as one more list** helps a little on top of either (+0.6 for decomposition with round robin) and every transformation chose it on the pool.

### Finding 3: behind the reranker, every gain disappears

| Transformation | No reranker | In front of the reranker |
|---|---|---|
| Rewrite | +0.2 (−0.7 to +1.3) | +0.4 (−0.2 to +0.9) |
| Multi-query | +0.4 (−0.7 to +1.4) | +0.5 (−0.4 to +1.4) |
| HyDE | +1.6 (−0.2 to +3.4) | +0.6 (−0.4 to +1.4) |
| Decomposition | +5.2 (+2.8 to +7.3) | +0.2 (−1.5 to +1.9) |
| Clauses (rule) | +1.5 (−0.9 to +4.3) | +0.6 (−0.9 to +2.0) |

- **The reason is in the last column of the reranker table.** Qwen3 reads the top 100 candidates, and those already hold 91.3% of the gold evidence. The transformations raise that to between 91.4% and 92.8%. Most of decomposition's gain without a reranker was in *ordering*: it raised recall@50 only from 84.5 to 87.0 while raising recall in 2,000 tokens by 5.2. Reordering is what the reranker does itself.
- **All five land within 0.4 points of each other** (82.6 to 82.9), the free rule among them. Nothing here separates a good transformation from a useless one once Qwen3 is behind it.
- This matches what the plan's research notes expected from a 2026 deployment study: gains from query rewriting and multi-query often vanish behind a reranker. Here it holds for decomposition and HyDE as well.
- **A measurement error was caught on the way.** The first version of this table showed every transformation at about +3. The script had reranked the reference row from a list cut at 50 candidates while the transformation rows passed 100, so the gain was depth, not transformation. Fixed in [scripts/stage07_tune.py](../../scripts/stage07_tune.py); the reference now reproduces the Stage 6 run exactly (82.3).

### Finding 4: Qwen3 does not need the question split, but it orders the top better when it is

The question deferred from Stage 6: MiniLM went from −12.9 to +3.0 when it scored each clause instead of the whole question. Qwen3 at depth 50, same candidates:

- **Recall in 2,000 tokens does not move**: +0.2 for the clauses, +0.8 (−0.4 to +1.8) for decomposition. The whole-question failure was MiniLM's: Qwen3 scoring the whole question was already 6.7 points above hybrid retrieval at this depth (Stage 6).
- **Decomposition does improve the top ten**: nDCG@10 +3.8 (+2.1 to +5.3) and full support@10 +4.8 (+1.8 to +7.5). Round robin over the sub-questions gives each fact a chunk near the top.
- **The rule's clauses do not** (+1.4 and 0.0). Fragments such as "despite being aware of safety concerns since 2014" are not questions a reranker can judge a chunk against.
- **On the pool the ordering was the same**: decomposition merged by round robin was the best split row (+1.1 recall, nDCG@10 63.3 against 60.2), and every clause row was slightly below Qwen3 alone.
- **It costs a reranking pass per part**: 3.26 passes per question, so about three times the reranker's time (Finding 5).

So the deferred question is closed: Qwen3 does not gain on the deciding metric from split questions. The top-ten gain is an open candidate for a short prompt, next to the listwise LLM reranker from Stage 6.

### Finding 5: what each transformation adds in time and money

Measured with every cache off on 20 development questions (8 for the reranker), on the M4 laptop. Per question, medians unless marked. Regenerate with `python scripts/stage07_sweep.py --timing` (about a cent).

| Transformation | LLM call, median | 95th percentile | Tokens in / out | Dollars per question | Searches | Retrieval |
|---|---|---|---|---|---|---|
| none | | | | | 1 | 0.21 s |
| Rewrite | 2.0 s | 2.4 s | 158 / 128 | $0.00008 | 2 | 0.43 s |
| Multi-query | 2.6 s | 5.0 s | 167 / 264 | $0.00015 | 4 | 0.91 s |
| HyDE | 3.1 s | 5.1 s | 184 / 287 | $0.00016 | 2 | 0.46 s |
| Decomposition | 3.1 s | 4.9 s | 189 / 247 | $0.00014 | 3.4 | 0.74 s |
| Clauses (rule) | none | | | $0 | 4.1 | 1.08 s |

| Qwen3 at depth 50 scores against | Passes per question | Reranking, median |
|---|---|---|
| the question | 1 | 5.7 s |
| the question and the decomposed sub-questions | 3.25 | 17.0 s |
| the question and the clauses | 3.62 | 19.1 s |

- **The LLM call is the cost that matters**: 2 to 3 seconds at the median and up to 5 at the 95th percentile, before retrieval can start. For 500 questions it is 4 to 8 cents at `gpt-6-luna`'s prices; most of the output is reasoning tokens.
- **Each extra search adds about 0.2 seconds**, almost all of it the embedding call, because the searches here run one after another. Embedding every query in one call, or in parallel, would bring that close to a single search.
- **Splitting for the reranker triples its time**: one pass per part, 17 to 19 seconds per question against 5.7 on this laptop. That is the price of decomposition's +3.8 nDCG@10.
- These are laptop numbers from a few questions; read them to one significant figure.

### Follow-up questions in the chat

[scripts/stage07_followups.py](../../scripts/stage07_followups.py) runs ten hand-written two-turn examples about articles in the corpus through the rewriter the app uses ([configs/chat.yaml](../../configs/chat.yaml): `gpt-6-luna`, the last three turns). It is a spot check, not a measurement: the corpus has no labelled conversations.

- **Eight lean on the earlier turn**, and all eight were rewritten with what the message left out. "When did they first beat them?" after a turn about Argentina and the All Blacks became "When did Argentina first beat the All Blacks?"
- **Two do not** (a new topic, and a question complete in itself), and both came back unchanged. Pulling the old topic into a new question is the usual way this step does harm.
- 10 of 10 passed the string check, for $0.0006. With no earlier turn the rewriter makes no call.
- In the app the follow-up is rewritten, the rewritten question is retrieved and answered, and the message as typed is shown above it.

### Did choosing on the pool cost anything?

| Transformation | Chosen on the pool | Pool gain | Development gain | Best form on development |
|---|---|---|---|---|
| Rewrite | both, round robin | +1.6 | +0.2 | both, fusion: +0.7 |
| Multi-query | both, round robin | +1.4 | +0.4 | the same, or queries alone: +0.4 |
| HyDE | both, round robin | +3.5 | +1.6 | the same |
| Decomposition | both, round robin | +5.2 | +5.2 | the same |
| Clauses (rule) | both, round robin | +2.0 | +1.5 | the same |

The pool picked the best development form for four of five, and the one it missed differs by half a point. The pool's gains are larger than development's for every transformation except decomposition, which is what choosing the best of several forms on one question set does.

### Framework equivalent

[framework_equivalents/query_transforms.py](../../framework_equivalents/query_transforms.py) wraps the hand-written pipeline's retriever and embedder for LangChain's `MultiQueryRetriever` and `HypotheticalDocumentEmbedder` (the `frameworks` extra, with `langchain-classic`).

An LLM's output is not deterministic, so a parity test cannot compare two live runs. [tests/test_query_parity.py](../../tests/test_query_parity.py) gives both sides the same scripted reply and compares what is deterministic:

- `test_multi_query_reads_the_same_queries_and_retrieves_the_same_chunks`
- `test_the_library_keeps_numbering_and_repeats_that_the_hand_written_parser_removes`
- `test_hyde_searches_with_the_same_vector_and_finds_the_same_chunks`

What differs:

- **Parsing.** The library keeps a reply's numbering ("1. Why was…") and repeated lines as queries, with no limit on how many. The hand-written parser strips, removes repeats and cuts, and counts each repair.
- **Combining.** `MultiQueryRetriever` returns the union of every query's results in the order first seen: all of the first query's, then what the second adds. There is no ranking across queries. Finding 2 shows that how lists are combined is worth 15 points here.
- **HyDE** is an embedder in LangChain: it replaces the dense query vector, averaging several passages if asked. In a hybrid pipeline the keyword side still needs a query of its own.
- **Failure.** Neither handles a refusal or an empty reply; the hand-written versions fall back to the question and count it.
- **LlamaIndex's `SubQuestionQueryEngine`** is not installed. It differs in kind: it answers each sub-question with its own retrieval and LLM call, then asks the LLM to combine the sub-answers. The hand-written decomposer stops at retrieval and gives one generator call the original question.

### New best config

**Unchanged: `rerank-qwen3-100`.** No transformation improved recall in 2,000 tokens behind the reranker, and each LLM transformation adds a call, seconds of latency and a failure mode to every question. `BEST_RUN` stays as it was.

- **Decomposition is the choice for a pipeline without a reranker**: [configs/stage07_query/decompose.yaml](../../configs/stage07_query/decompose.yaml), +5.2 against hybrid retrieval, for one `gpt-6-luna` call per question.
- **No answer-quality run was made.** The retrieval result behind the reranker is level, so an answer run would compare two pipelines that hand the generator the same evidence.

**Open items carried forward:** human labels for the judge (before Stage 8); a gold-evidence row with source and date headers (with Stage 8); the 64-token chunk candidate (Stage 8); for a prompt of five to ten chunks, the listwise LLM reranker (Stage 6) and now the reranker scoring decomposed questions (+3.8 nDCG@10 at depth 50) (Stage 8); the title, source and date header in the retrieval indexes (A1, or Stage 8). Closed in this stage: whether Qwen3-Reranker gains from split questions (not on recall within 2,000 tokens).

### Cost

Stage 7 spent $0.49, taking the project to $8.13 of $40:

| Item | Cost |
|---|---|
| Five example outputs per transformation (the spot check) | $0.003 |
| Transforming 784 pool and development questions, four LLM transformations | $0.47 (estimate $0.47) |
| Embedding the transformed queries | $0.005 |
| Latency measurement, caches off | $0.011 |
| Follow-up spot check | $0.0006 |
| Development runs (everything cached) | under $0.001 |

- The local reranker was free: about 200,000 new pairs over 7 hours of laptop time.
- The first estimate for transforming the questions was $1.27; it assumed 650 reasoning tokens per reply. Measured on the spot check, the replies were 130 to 490 tokens, and the corrected estimate matched the bill.
- No answer-quality run was made (see "New best config").

## When you would choose differently in production

- **With no reranker, decompose multi-part questions**, and combine the sub-questions' lists by round robin. Here that was worth 5 points for one cheap LLM call. Fusion would have cost 10.
- **With a good reranker, measure before adding a transformation.** Here none of five survived it. The reranker already reads deep enough to hold almost all the evidence; a transformation that only reorders the first stage has nothing left to do.
- **Transform when the first stage cannot reach the evidence at any depth.** That is where a transformation adds what a reranker cannot: vocabulary the user does not share with the documents, abbreviations, or questions whose wording matches nothing in the corpus. These news questions repeat the articles' own words, which is why rewriting found nothing new.
- **Be careful with HyDE on a private corpus.** It writes an answer from the model's memory. On public news that answer is often right, which flatters the method; on internal documents it is a confident guess, and a wrong name is a query for the wrong documents.
- **Rewrite follow-ups in any chat.** That is not an optimisation: "When did they first beat them?" retrieves nothing useful as typed. Check that a new topic comes back unchanged.
- **Plan for the LLM step failing.** Fall back to the question, count it, and watch the count. Here the cheapest model sometimes spent its whole output limit reasoning and returned nothing.
- **Count the latency where the user waits.** One LLM call before retrieval sits on the critical path of every question, and a reranking pass per sub-question multiplies the reranker's time.

## Questions you should now be able to answer

1. Why does HyDE embed a made-up answer instead of the question, and what is the risk?
2. Why did decomposition help and multi-query not, without a reranker?
3. Why did reciprocal rank fusion turn decomposition from a gain into a loss?
4. Why did every transformation's gain disappear behind Qwen3?
5. Did Qwen3-Reranker gain from split questions as MiniLM did? Why not?
6. What does query transformation cost per question, and where does the time go?
7. Why is follow-up rewriting kept out of the results tables?

<details>
<summary>Short answers</summary>

1. A passage that answers the question looks more like the chunk holding the answer than the question does, so their vectors are closer. The passage is written from the model's memory: here it named "Uber" and "Huawei", and on a corpus the model has not seen it would name a guess.
2. Each sub-question is about one fact, so it matches the one chunk that states that fact. The paraphrases kept every detail of the question in a new order, and found what the question had found.
3. Fusion scores a chunk by how many lists rank it well. Sub-questions about different facts should find different chunks; the chunk they all find is the one that mentions every name and states none of the facts. Round robin gives each sub-question its place instead.
4. The reranker reads 100 candidates that already hold 91.3% of the evidence, and the transformations raise that by at most 1.5 points. What they did without a reranker was reorder the list, and the reranker reorders it anyway.
5. No: +0.8 at most on recall within 2,000 tokens. MiniLM failed when it judged a chunk against the whole question; Qwen3 judging the whole question was already well above hybrid retrieval. Decomposed questions did improve Qwen3's top ten (+3.8 nDCG@10).
6. About 2 to 3 seconds and $0.0001 to $0.0002 for the LLM call, which comes before retrieval; about 0.2 seconds per extra search; and for split reranking about three times the reranker's time (17 s against 5.7 s here).
7. The evaluation questions are single-turn, and there are no labelled conversations in the corpus. Ten hand-written examples are a check that the step works, not a measurement of how well.

</details>
