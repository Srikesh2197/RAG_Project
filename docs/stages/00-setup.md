# Stage 0: Setup and data

No retrieval yet. This stage fixes the three things every later number depends on: the corpus, the gold labels, and the split between questions we tune on and questions we hold back.

## Concept

**What an evaluation set for RAG needs.** Four things: a corpus, questions, a reference answer per question, and gold evidence saying where in the corpus the support for that answer is.

Gold evidence is what lets you score retrieval separately from generation. When an answer is wrong, you can then tell whether the evidence was never retrieved or was retrieved and misused. Those two failures have different fixes, and a pipeline that only measures final answers cannot tell them apart.

**Why gold evidence is stored as character spans.** A chunk id is an output of the chunker. Change the chunk size and every id points somewhere else, so labels written as chunk ids die at the first chunking experiment. A span `(doc_id, start_char, end_char)` in the unchunked article depends on nothing downstream. Any chunk from any chunker is relevant if it overlaps a gold span.

For this to work, the article text has to be frozen. `Document.text` is the canonical text and nothing edits it after ingestion.

## Options

**Label granularity**

| Label | Survives a chunker change | Tells you the right part was retrieved | Cost to produce |
|---|---|---|---|
| Chunk id | no | yes, for one chunker only | low |
| Document id | yes | no | low |
| Character span | yes | yes | needs the evidence text to be locatable in the source |

**Corpus.** The candidates and why MultiHop-RAG was chosen are in [research-notes.md](../research-notes.md#12-corpus-candidates).

**Split**

| Split | Dev and test have the same mix | Test evidence is unseen in dev |
|---|---|---|
| Random, stratified by question type | yes | no |
| Evidence-disjoint | no | yes |

## What we chose

- **Corpus:** MultiHop-RAG, pinned to one dataset revision in [configs/data.yaml](../../configs/data.yaml). The manifest records the SHA-256 of both raw files.
- **Ids:** `doc_0000`… and `q_0000`… in file order, which the pinned revision makes stable.
- **Canonical text:** the article body. Title, source, date, category and URL go in metadata.
- **Span mapping:** exact string match, with a fallback that ignores whitespace differences. A question with any unmapped evidence would be excluded from the splits.
- **Split:** evidence-disjoint, 500 development and 500 test questions, described under "Finding 2" below.
- **Download:** `huggingface_hub` at the pinned revision instead of the `datasets` library. The dataset is two JSON files, and this avoids a heavy dependency.
- **Python:** 3.14. Every library planned for later stages has a 3.14 wheel for this Mac except `hnswlib`, so Stage 11 will use FAISS's HNSW index.

## Measured result

Full output: [results/stage00_data_profile.md](../../results/stage00_data_profile.md). Regenerate with `make profile`.

| | |
|---|---|
| Articles | 609, from 49 sources, published 26 Sep to 25 Dec 2023 |
| Corpus size | 1,394,467 tokens |
| Article length in tokens | median 1,686; 90th percentile 3,811; max 16,133 |
| Questions | 2,556: comparison 856, inference 816, temporal 583, unanswerable 301 |
| Evidence per answerable question | 2 to 4 sentences, always from at least 2 articles |
| Evidence sentence length | median 34 tokens |
| Gold spans as a share of the corpus | 2.4% of characters |

One example of each type, from the development pool:

| Type | Question (shortened) | Answer | What retrieval has to do |
|---|---|---|---|
| inference | Who is the individual facing a criminal trial on fraud charges, as reported by both The Verge and TechCrunch…? | Sam Bankman-Fried | Find sentences in several articles that describe the same unnamed entity. |
| comparison | Does 'The Age' article suggest Australia's Davis Cup team is aiming to improve, while the 'Sporting News' article indicates South Africa's rugby team already has? | Yes | Find one sentence per side; the two sides share almost no vocabulary. |
| temporal | Has the advice from Sporting News to bettors… between the reports published on September 28, 2023, and December 18, 2023? | no | Find sentences in specific articles identified by source and date. |
| null | Considering forecasts from a Bloomberg article and discoveries reported by Al Jazeera, which country…? | Insufficient information. | Find nothing sufficient, so the generator must decline. |

Many questions name their sources and dates. That is why metadata filtering (Stage 11) and source and date headers in the prompt (Stage 8) have something to measure here.

### Finding 1: every evidence sentence maps to a span

All 6,084 evidence items were found verbatim in their article. The whitespace fallback was never needed and no question was excluded.

One piece of label noise: 29 evidence sentences occur more than once in their article. The span marks the first occurrence, so a chunk holding only a later copy would be scored as a miss. That is 0.5% of evidence items.

### Finding 2: questions share evidence heavily, which changed the split

The 6,084 evidence items are only 981 distinct sentences. One TechCrunch sentence about Sam Bankman-Fried is gold for 437 questions and one about Google for 376. Linking questions that share a sentence gives 303 clusters, and the two largest hold 481 and 410 of the 2,255 answerable questions.

This matters in two ways.

- **A random split leaks.** With the random split in the original plan, 286 of 441 answerable test questions had all their gold sentences already gold in the development set. A held-out test like that cannot show whether tuning overfit to particular evidence.
- **Questions are not independent.** Results for questions that share evidence move together, so a confidence interval that treats 500 questions as 500 independent samples is too narrow.

What we did:

1. Whole clusters are assigned to the development side or the test side, so the two never share a gold sentence. The manifest records the check: 0 shared.
2. No cluster may supply more than 20% of a split's answerable questions. Larger clusters are randomly thinned. The largest cluster ends up at 12.9% of development and 10.9% of test.
3. Each split is a type-stratified sample from its side, plus 59 unanswerable questions.
4. Every question carries its `cluster` id. Stage 2 will compute confidence intervals by resampling clusters.

The cost is that the two splits differ in make-up. The Bankman-Fried cluster is in development and the Google cluster is in test, and development has 105 inference questions against 130 in test. A drop from development to test at Stage 12 could therefore be overfitting or just the different mix. To separate them, Stage 12 will also run the untuned baseline on test: the baseline's gap measures the mix, and any extra gap for the tuned pipeline is overfitting.

Leftover questions go to two files. `questions_pool.jsonl` (870 questions) shares evidence only with development, so it is free to use for prompt examples and judge calibration. `questions_test_pool.jsonl` (686) shares evidence with test and stays untouched.

### Finding 3: answers are easy to guess

52.5% of reference answers are "yes" or "no", 11.8% are "Insufficient information", and "Sam Bankman-Fried" and "Google" are another 18.9%.

A guesser that reads only the question's type and gives that type's most common answer would score **54.6% on development and 51.8% on test**. So the floor for answer correctness is about 55%, not 0%. The paper reports 89% for GPT-4 given the gold evidence, which leaves roughly 35 points of usable range.

Consequences for Stage 2:

- The results table gets a "guesser" row next to the no-retrieval and gold-evidence rows.
- Answer correctness is reported per question type as well as overall.
- Retrieval metrics are the sharper instrument on this dataset. Answer metrics still matter for the stages that change generation (8 and 9).

### Finding 4: shape of the corpus

- The median article is about 1,700 tokens, so about three or four chunks at 512 tokens. The longest is 16,133.
- Gold evidence is 2.4% of the text. Retrieval is looking for a few sentences among many chunks that are on-topic and not evidence.
- Evidence is spread through the article with a tilt to the opening: 29.5% of evidence starts in the first fifth, and 15–19% in each later fifth.
- Every one of the 609 articles holds some gold evidence. There are no pure distractor documents.
- One full index build with `text-embedding-3-small` costs about 3 cents (1.39M tokens at $0.02 per million).

## When you would choose differently in production

- **You would not have this dataset.** You would build one from real user questions and label the evidence by hand. Fifty to a hundred questions chosen to cover your real failure cases are worth more than thousands of generated ones. Generated test sets rank retrievers reasonably and rank generators poorly (see the research notes).
- **Label at document level** if labelling time is short and you do not plan to tune chunking. Label spans if you do.
- **Split by time** if the corpus grows, so the test questions are about documents newer than anything you tuned on. Split by customer or document family if those are your units of generalisation.
- **Real corpora need the parsing step first** (Stage 10), plus deduplication and a rule for what happens to labels when a document is revised.

## Questions you should now be able to answer

1. Why can't gold labels be chunk ids?
2. What does each of the four question types demand from retrieval?
3. Why does shared evidence between development and test weaken a held-out test, and what does the cluster cap add on top of keeping them disjoint?
4. Why is 55%, not 0%, the number to beat on answer correctness?
5. Why pin the dataset revision and record file hashes?

<details>
<summary>Short answers</summary>

1. Chunk ids are produced by the chunker, so they change whenever the chunker does. A character span in the unchunked text does not.
2. Inference: find several sentences about one unnamed entity. Comparison: find evidence for each side, which usually shares little wording. Temporal: find evidence in specific sources and dates. Null: find nothing sufficient, so the generator declines.
3. If test questions rely on sentences that development questions also rely on, a configuration tuned to retrieve those sentences looks good on both, and the test tells you nothing new. The cap stops one sentence's cluster from dominating a split, so no single chunk decides a large share of the score.
4. Because of the answer distribution: always giving each question type's most common answer already scores about 55%.
5. So that every clone builds byte-identical data and every result in the repo can be traced to exactly the inputs that produced it.

</details>
