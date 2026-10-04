# Stage 0 data profile

Dataset `yixuantt/MultiHopRAG` at revision `71ac0d0bd1f9`. Seed 42.

## Documents

- 609 articles, 1,394,467 tokens in total.
- 49 sources. Largest: Sporting News (101), TechCrunch (97), The Verge (45), Polygon (44), The Guardian (29).
- Categories: sports (211), technology (172), entertainment (114), business (81), science (21), health (10).
- Published between 2023-09-26 and 2023-12-25.

| Length per article | min | median | 90th percentile | max |
|---|---|---|---|---|
| characters | 4770 | 7803 | 16897 | 71034 |
| tokens | 1057 | 1686 | 3811 | 16133 |

## Questions by split and type

| type | dev | test | pool | test_pool |
|---|---|---|---|---|
| comparison_query | 211 | 196 | 234 | 215 |
| inference_query | 105 | 130 | 298 | 283 |
| null_query | 59 | 59 | 183 | 0 |
| temporal_query | 125 | 115 | 155 | 188 |
| total | 500 | 500 | 870 | 686 |

Answer-quality subset inside dev: 150 questions (comparison_query 63, inference_query 32, null_query 18, temporal_query 37).

## Reference answers

All 2,556 questions:

| form | questions | share |
|---|---|---|
| yes / no | 1343 | 52.5% |
| entity or phrase | 912 | 35.7% |
| insufficient information | 301 | 11.8% |

Most common answers: Yes (30.6%), no (21.0%), Insufficient information. (11.8%), Sam Bankman-Fried (10.6%), Google (8.3%), Sam Altman (2.2%).

Most common answer per question type, and the score from always giving it (a guesser that reads only the question's type):

| split | comparison_query | inference_query | null_query | temporal_query | guesser's score |
|---|---|---|---|---|---|
| dev | yes (57%) | sam bankman-fried (33%) | insufficient information. (100%) | no (47%) | 54.6% |
| test | yes (56%) | sam altman (28%) | insufficient information. (100%) | yes (46%) | 51.8% |

## Gold evidence

- Mapped to character spans: 6,084 of 6,084 (100.00%), 6,084 verbatim.
- Evidence sentences that occur more than once in their article: 29 (the span marks the first occurrence).
- Questions with no evidence (unanswerable): 301.
- Articles holding at least one gold span: 609 of 609.
- Gold spans cover 2.4% of all corpus characters (981 distinct spans).
- Evidence sentence length in tokens: median 34, 90th percentile 51, max 102.

| evidence items per answerable question | questions | share |
|---|---|---|
| 2 | 1079 | 47.8% |
| 3 | 778 | 34.5% |
| 4 | 398 | 17.6% |

| distinct articles per answerable question | questions | share |
|---|---|---|
| 2 | 1169 | 51.8% |
| 3 | 774 | 34.3% |
| 4 | 312 | 13.8% |

| where in the article the evidence starts | share of evidence |
|---|---|
| 0–20% of the way through | 29.5% |
| 20–40% of the way through | 18.0% |
| 40–60% of the way through | 18.2% |
| 60–80% of the way through | 15.3% |
| 80–100% of the way through | 19.0% |

## Evidence shared between questions

- 6,084 evidence items are 981 distinct sentences. A sentence is gold for 2 questions at the median and 437 at most; 269 are used once.
- Linking questions that share a sentence gives 303 clusters. The largest hold 481, 410, 81, 76 questions; 85 clusters are a single question.
- Gold sentences shared by dev and test: 0. Shared by pool and test: 0.

| questions using it | source | most reused gold sentences |
|---|---|---|
| 437 | TechCrunch | The prosecution painted Bankman-Fried as someone who knowingly committed fraud to achieve … |
| 376 | TechCrunch | The case, filed by Arkansas-based publisher Helena World Chronicle, argues that Google “si… |
| 88 | TechCrunch | The second week of the trial’s standout testimony came from Alameda Research’s former CEO … |

| split | answerable questions | clusters represented | largest cluster's share |
|---|---|---|---|
| dev | 441 | 120 | 12.9% |
| test | 441 | 116 | 10.9% |
