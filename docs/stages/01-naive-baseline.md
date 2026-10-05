# Stage 1: Naive baseline and Streamlit app

The simplest pipeline that works end to end. Every later stage changes one thing in it and reports the difference against it, so its job is to be plain, not good.

## Concept

**The path from document to answer.**

| When | Step | What it does | Code |
|---|---|---|---|
| Index time (once) | Chunk | Cut each article into pieces of 512 tokens. | [chunking/fixed.py](../../src/ragbasics/chunking/fixed.py) |
| | Embed | Turn each chunk into a vector of 1,536 numbers. | [embedding/openai_embedder.py](../../src/ragbasics/embedding/openai_embedder.py) |
| | Store | Keep the vectors in one matrix with the chunk text alongside, saved to disk. | [index/numpy_store.py](../../src/ragbasics/index/numpy_store.py) |
| Query time (per question) | Embed | Turn the question into a vector with the same model. | |
| | Search | Score every chunk against the question and keep the top 5. | [index/numpy_store.py](../../src/ragbasics/index/numpy_store.py) |
| | Prompt | Paste those chunks and the question into a prompt. | [context/prompts.py](../../src/ragbasics/context/prompts.py) |
| | Generate | The LLM answers from the pasted text. | [generation/llm.py](../../src/ragbasics/generation/llm.py) |

[pipeline.py](../../src/ragbasics/pipeline.py) wires these together and returns a `Trace`: the retrieved chunks with scores and ranks, the exact prompt, the answer, and the time, tokens and cost of each step.

**Why chunk at all.**

- **Dilution.** An embedding is one vector for the whole input. A 1,700-token article covers several topics, and its vector averages them, so one evidence sentence (median 34 tokens here) barely moves it.
- **Model limit.** `text-embedding-3-small` accepts at most 8,191 tokens. The longest article is 16,133.
- **Prompt cost and distraction.** The generator should read the few relevant passages, not five whole articles.

**Cosine similarity against dot product.** The dot product of two vectors grows with their lengths as well as with how closely they point the same way. Cosine similarity divides the lengths out, so only direction counts. If every vector is scaled to length 1 before it is stored, the two are the same number, and scoring a question against the whole index is one matrix-vector product: `scores = vectors @ query`. `test_a_long_vector_wins_on_dot_product_and_loses_on_cosine` shows the difference on two vectors.

OpenAI's embedding models already return vectors of length 1. The store scales them anyway, because not every model does (Stage 4 uses some that do not).

**A score has no absolute meaning.** Each model spreads its scores over its own range, so a cosine of 0.4 can be the best match in one model and a poor one in another. Only the ranking within one model and one index means something.

**The question and the index must use the same model.** Two models place text in unrelated spaces, so a score between a vector from one and a vector from the other is noise. If the dimensions differ you get an error. If they happen to match, you get results that look normal and are wrong. The index therefore records the chunker and embedder that built it in `meta.json`, and the pipeline refuses to load it under a different config.

**The grounded-answer prompt** has four parts:

1. an instruction to answer only from the supplied passages;
2. what to say when the passages do not contain the answer ("Insufficient information.", which is also the dataset's reference answer for unanswerable questions);
3. the passages, numbered, in rank order;
4. the question.

## Options

| Decision | Baseline | Alternatives, and where they are measured |
|---|---|---|
| Chunk boundary | every 512 tokens | sentence, paragraph, recursive, semantic, parent-child (Stage 3) |
| Overlap | none | 10% and 20% (Stage 3) |
| Embedder | `text-embedding-3-small` | larger and local models (Stage 4) |
| Search | exact, dense only | keyword and hybrid (Stage 5), approximate (Stage 11) |
| Chunks in the prompt | top 5, rank order, no headers | 3 to 20, reordered, with source and date headers (Stage 8) |
| Abstention | one sentence in the prompt | sufficiency check, score threshold (Stage 9) |

## What we chose

Everything is in [configs/baseline.yaml](../../configs/baseline.yaml).

- **Chunker:** 512 tokens of `cl100k_base` (the tokenizer the embedder uses), no overlap. Each chunk is a slice of the article and keeps its `start_char` and `end_char`, so Stage 2 can check it against the gold spans.
- **Embedder:** `text-embedding-3-small`, 100 chunks per request.
- **Store:** a NumPy matrix. Files on disk: `vectors.npy`, `chunks.jsonl`, `meta.json`.
- **Search:** exact. With 3,036 chunks it takes about a millisecond.
- **Generator:** `claude-sonnet-5-5`, with `effort: low`. This model reasons before answering and the reasoning is billed as output tokens; `low` keeps that small. It accepts no temperature setting, so answers can vary a little between runs. Stage 2 measures that variation.
- **Two indexes:** the corpus in `data/index/baseline`, uploaded files in `data/index/baseline_uploads`.
- **Offline config:** [configs/offline.yaml](../../configs/offline.yaml) swaps in a word-hashing embedder and no LLM, so the tests, CI and the app run without keys. It is not used for any measurement.

Three things the baseline deliberately leaves out, because later stages measure them: source and date headers on passages, any use of the question's named sources, and memory of earlier chat turns.

## Measured result

There is no comparison in this stage. What can be measured before Stage 2:

| | |
|---|---|
| Chunks | 3,036 from 609 articles (median 4 per article, max 32) |
| Chunk size | 512 tokens for all but each article's last chunk; 109 chunks are under 100 tokens |
| Development gold sentences cut in two by a chunk boundary | 19 of 387 (4.9%) |
| Tokens to embed the corpus | 1,394,468, about $0.03 |

Regenerate with `python scripts/profile_chunks.py`.

Two notes on these numbers:

- Two chunks measure 513–514 tokens when re-tokenised alone. A slice of text can tokenise slightly differently from the same characters inside the full article.
- A sentence cut in two is not lost: each half is still in a chunk. But neither chunk holds the full sentence, so the generator sees half of it unless both chunks are retrieved.

### Observations from 10 questions in the app

Ten questions are an illustration, not a measurement. Stage 2 runs all 500.

**Predictions, written before trying the app**

1. Question type that fails most: temporal.
2. Main cause, retrieval or generation: retrieval does not find the right passages.

The ten development questions asked in the app, with their reference answers.

| # | Type | Question | Reference answer |
|---|---|---|---|
| 1 | inference | Who is the individual under 30, once considered the richest in that age group according to TechCrunch, who is now facing a criminal trial on charges including fraud and conspiracy, and has pleaded not guilty, with allegations of using fraudulent means for gaining wealth, power, and influence? | Sam Bankman-Fried |
| 2 | inference | What is the name of the company that was discussed on TechCrunch for removing AI-created songs and introducing an AI-powered DJ feature, and was also mentioned on The Verge for achieving its first operating profit in a year, leading to a significant rise in its stock value? | Spotify |
| 3 | inference | Which group of individuals can take advantage of hype around specific events to make money by placing wagers on scenarios such as whether a team will lead at the end of a certain quarter, as reported by Sporting News? | Bettors |
| 4 | comparison | Do 'The Verge' and 'Engadget' articles both suggest that 'Consumers' have guides or opportunities to make better purchasing decisions, while 'TechCrunch' discusses 'Consumers' desire for a new model in a different sector? | Yes |
| 5 | comparison | Does the TechCrunch article on GPT-4 suggest a greater ease of prompting toxic output compared to other models, while the TechCrunch article on Meta's open source AI approach indicate concerns of potential danger and disinformation from industry competitors like Google, OpenAI, and Microsoft? | Yes |
| 6 | comparison | Does the TechCrunch article suggest that the success in "North America's EV market" is due to the size and price of electric vehicles, while The Verge article focuses on Donald Trump's criticism of electric vehicles regarding their cost, range, and impact on American jobs? | no |
| 7 | temporal | Has the approach of Sportsbooks in adjusting betting lines and odds, as reported by Sporting News after October 4, 2023, and before November 1, 2023, remained consistent? | no |
| 8 | temporal | After the Polygon report on Valve's updates to the Steam Deck hardware published on November 9, 2023, and the Engadget review of the Steam Deck OLED version published on the same date, was the reporting on Valve's improvements to the Steam Deck hardware consistent? | Yes |
| 9 | null | Considering the information from an article by The Guardian and another by Forbes about Oleg Fomenko, which company did he found that is mentioned in both articles and also received significant investment from a major venture capital firm as reported by The Guardian? | Insufficient information. |
| 10 | null | Considering the information from an article in The Economic Times about Dunzo's latest funding round and a piece from Business Standard detailing Dunzo's expansion plans into new cities, which city, starting with the letter 'B', is both a location where Dunzo has recently expanded its services and is also the city where one of its new investors is headquartered? | Insufficient information. |

**What happened.** The first three columns are the notes taken in the app. The last column was added afterwards by checking each gold evidence sentence against the ranked list: it gives the rank of the chunk that contains each one.

| # | Type | Answer | Note from the app | Rank of each gold sentence |
|---|---|---|---|---|
| 1 | inference | right | Sources appear. | 12, 12, 13, 49 |
| 2 | inference | right, with a caveat | Answered Spotify, and said the passages only covered the second half of the question. | 2, beyond 50 |
| 3 | inference | wrong: "Insufficient information." | Sporting News is among the sources, so this looked like a generation failure. | 23, 48 |
| 4 | comparison | wrong: "Insufficient information." | Named sources not retrieved. | all three beyond 50 |
| 5 | comparison | wrong: "Insufficient information." | Named sources not retrieved. | 28, 45 |
| 6 | comparison | wrong: said yes | Sources appear. | 1, 4 |
| 7 | temporal | wrong: "Insufficient information." | Retrieved chunks were dated outside the range in the question. | 20, 30 |
| 8 | temporal | right | Sources appear. | 1, 6, 24 |
| 9 | null | right | Nothing relevant retrieved, as expected. | no gold |
| 10 | null | right | Nothing relevant retrieved, as expected. | no gold |

**Score:** 5 of 10 right. Inference 2 of 3, comparison 0 of 3, temporal 1 of 2, null 2 of 2.

**Against the predictions**

- **Main cause: retrieval. Supported.** In 4 of the 5 wrong answers (3, 4, 5, 7) no gold sentence was in the top 5. Only question 6 had its evidence retrieved and was still answered wrongly.
- **Type that fails most: temporal. Not shown here.** Comparison failed 3 of 3 and temporal 1 of 2. With two or three questions per type this says little either way.

**What the ten questions show**

- **The right source is not the right passage.** In question 3 the top 5 held four Sporting News chunks, and none contained the evidence, which sat at ranks 23 and 48. In questions 5 and 7 the correct article was in the top 5 and the wrong chunk of it was retrieved. Judging retrieval by source name overstates it, which is why Stage 2 scores against gold spans.
- **The evidence is usually close.** In 4 of the 8 answerable questions every gold sentence was outside the top 5 and at least one was inside the top 50. A second-stage reranker works on exactly that range (Stage 6).
- **Multi-hop questions need every piece.** Only question 6 had all its gold sentences in the top 5. Question 2 had one of two, and the model said so.
- **A retrieval miss became an abstention, not a wrong guess.** All four retrieval failures were answered "Insufficient information." The abstention line in the prompt is doing that. The cost is that a miss and a truly unanswerable question look the same from the answer alone.
- **A right answer does not prove retrieval worked.** Question 1 was answered correctly with no gold sentence in the top 5. Sam Bankman-Fried's trial is covered by many articles, and other chunks about it were enough. Question 8 was right with one of three in the top 5.
- **Dates are invisible to the search.** The chunk text does not carry the article's date or source, so "after October 4 and before November 1" cannot steer retrieval. In question 7 three of the five chunks came from articles dated outside that range.
- **Answers are not always bare.** Question 2 came back as three sentences. Stage 2's answer scoring has to cope with that.

## When you would choose differently in production

- **Never ship fixed-size chunks as the final choice.** They are the control. At minimum, split on paragraph and sentence boundaries (Stage 3).
- **A NumPy matrix is fine up to roughly 50–100 thousand vectors** on one machine. Beyond that, or with several processes writing, or when you need filtering and deletes, use a vector store (Stage 11).
- **Re-ingestion needs more than skipping known ids.** This pipeline skips a document whose id is already indexed and cannot update or delete one. Production ingestion hashes content, replaces changed documents and removes deleted ones (Stage 10).
- **Changing the embedding model means re-embedding everything.** Plan for it: keep the canonical text, and record the model with the index as `meta.json` does.
- **A hosted embedding API sends your documents to a vendor.** If that is not allowed, use a local model (Stage 4).
- **Chat needs conversation handling.** A follow-up such as "and what about Google?" retrieves nothing useful on its own (Stage 7).

## Questions you should now be able to answer

1. Why chunk at all?
2. Does a similarity score of 0.4 mean anything on its own?
3. Why are the vectors scaled to length 1?
4. What breaks if the question and the index use different embedding models?
5. What are the four parts of a grounded-answer prompt?
6. Why do chunks keep `start_char` and `end_char`?

<details>
<summary>Short answers</summary>

1. One vector for a long document averages its topics, so a single relevant sentence is diluted; the embedder has a token limit; and the generator should read only the relevant passages.
2. No. Score ranges differ between models, so only the ranking within one model and index is meaningful.
3. So that the dot product equals cosine similarity and a long vector cannot win on length alone. Search is then one matrix-vector product.
4. The two models' spaces are unrelated. Different dimensions raise an error; equal dimensions give normal-looking results that are noise.
5. Answer only from the passages; what to say when they fall short; the numbered passages; the question.
6. So any chunk from any chunker can be compared with the gold evidence spans, which are character ranges in the same unchunked text.

</details>
