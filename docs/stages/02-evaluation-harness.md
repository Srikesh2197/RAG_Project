# Stage 2: Evaluation harness

Nothing in the pipeline changes in this stage. It builds the instruments every later stage is read with, and uses them on the Stage 1 baseline and on three reference rows.

## Concept

**Two layers, scored separately.** Retrieval is scored against the gold evidence spans. Answers are scored against the reference answer. Scoring only the answer hides which half failed, and the two halves have different fixes.

**Retrieval metrics.** A chunk is relevant if it overlaps a gold span. Code: [eval/retrieval_metrics.py](../../src/ragbasics/eval/retrieval_metrics.py).

| Metric | What it asks | What it ignores |
|---|---|---|
| Evidence recall@k | What share of the question's gold sentences are in the top k chunks? | Order inside the top k |
| Full-support@k | Are all of them there? (0 or 1) | Partial progress |
| MRR@10 | How high is the first relevant chunk? (1/rank) | Every relevant chunk after the first |
| nDCG@10 | Are relevant chunks near the top? Each adds 1/log2(rank+1); the sum is divided by the best possible ordering. | Nothing, but it is harder to read |
| Recall in 2,000 tokens | Recall when the budget is an amount of text, not a number of chunks | |
| Precision in 2,000 tokens | What share of that text is gold evidence? | |

- **Recall and MRR disagree on multi-hop questions.** One gold sentence at rank 1 and the other never retrieved gives MRR 1.0 and full-support 0. `test_mrr_and_full_support_can_disagree` shows it.
- **Recall@5 is unfair across chunk sizes.** Five 1,024-token chunks hold eight times the text of five 128-token chunks. The token-budget metrics read the ranked list until 2,000 tokens are used and cut the last chunk at the budget, so every chunker gets the same amount of text. Stage 3 depends on this.

**Answer metrics.** Code: [eval/answer_metrics.py](../../src/ragbasics/eval/answer_metrics.py) and [eval/judge.py](../../src/ragbasics/eval/judge.py).

- **Correctness is a cascade.** A normalised string comparison settles bare answers for free. A bare yes, no or abstention that differs from the reference is settled as wrong. Everything else, such as a three-sentence answer, goes to an LLM judge.
- **Faithfulness** asks whether the answer follows from the passages in the prompt. It is separate from correctness: an answer recalled from memory can be correct and unfaithful.
- **Abstention has two sides.** Declining on a null question is right. Declining on an answerable one is a false abstention, and it is split by whether the evidence was in the prompt.

**The judge is a model with an error rate.** It is checked three ways: on every answer the string comparison already settled, against a second and larger judge, and against human labels.

**How big must a difference be.** Code: [eval/stats.py](../../src/ragbasics/eval/stats.py).

- A bootstrap redraws the questions many times and reads the spread of the metric.
- Questions here share evidence, so whole evidence clusters are redrawn. Redrawing single questions would treat correlated results as independent.
- Two configs are compared on the same questions. Pairing cancels question difficulty, so a small real difference can be seen where two separate intervals would overlap.
- The generator has no temperature setting, so one repeated run measures how much the answers move when nothing changes.

## Options

| Decision | Options | Trade-off |
|---|---|---|
| Relevance label | chunk overlaps a gold span; chunk contains the whole span | Overlap credits half a sentence. Containment penalises a chunker for a boundary it could not see. |
| Budget for retrieval | fixed k; fixed tokens | Fixed k is what the prompt uses. Fixed tokens is comparable across chunk sizes. |
| Answer correctness | exact match; token F1; LLM judge; cascade | Exact match fails every long answer. F1 is poor on yes/no. A judge costs money and makes errors. |
| Judge output | binary; 1–5 scale | Binary verdicts are easier to check against a person and to turn into a rate. |
| Faithfulness | one verdict per answer; share of supported claims (RAGAS, DeepEval) | Claim-level is finer and depends on how the answer is split into claims. |
| Interval | question bootstrap; cluster bootstrap | The cluster version is wider and honest when questions share evidence. |

## What we chose

- **Config:** [configs/eval.yaml](../../configs/eval.yaml), kept apart from the pipeline configs so a stage changes the pipeline and never the measurement.
- **Retrieval:** 50 chunks retrieved per question, scored on all 441 development questions that have gold evidence. Relevance is overlap.
- **Answers:** the fixed 150-question subset (63 comparison, 37 temporal, 32 inference, 18 null).
- **Judge:** `gpt-6-luna`, a different vendor from the generator. It costs about 3 cents per run. `gpt-6.1-sol` costs twenty times more per token and was used only as a comparison on the labelled sample.
- **Intervals:** 95%, from 2,000 redraws of evidence clusters.
- **Cache:** every generator and judge call goes through a SQLite cache keyed by model, settings and prompts. A repeat run uses `--sample 1`, which is part of the key.
- **Cost gate:** `rag eval` prints an estimate first. A run above $1 needs `--yes`, and `--estimate` prints the cost and stops.
- **Runs:** each run writes `metrics.json`, `config.yaml` and `questions.jsonl` to `runs/<id>/`. `rag report` rebuilds [results/results.csv](../../results/results.csv) and the README table; `rag compare a b` gives paired differences.
- **Test split:** the runner refuses the held-out files.

## Measured result

No predictions were written down for this stage.

**Open items carried forward:** human labels for the judge (before Stage 8), and a gold-evidence row with source and date headers (with Stage 8).

### Retrieval on the baseline

441 questions. Percentages with 95% intervals.

| recall@5 | full support@5 | recall@10 | full support@10 | MRR@10 | nDCG@10 | recall in 2,000 tok | precision in 2,000 tok |
|---|---|---|---|---|---|---|---|
| 48.2 (41–57) | 21.5 (15–32) | 62.6 | 36.5 | 59.0 (53–66) | 49.0 (43–56) | 44.8 (38–53) | 1.7 (1.5–1.9) |

- The first relevant chunk is usually near the top (MRR 59), and all the evidence is in the prompt for only about one question in five. That gap is the multi-hop problem.
- 98% of the text sent to the generator is not gold evidence.
- By type, recall@5 is 55% for comparison, 52% for temporal and 30% for inference. Full support@5 for inference is 4%.

### Answers: the baseline and three reference rows

150 questions.

| Run | Correct | Correct, answerable | Abstains on null | False abstentions | Correct vs baseline |
|---|---|---|---|---|---|
| guesser | 52.7 (40–67) | 46.2 | 100 | 0 | +1.3 (−13 to +17) |
| baseline | 51.3 (42–60) | 44.7 | 100 | 52.3 (43–63) | |
| baseline, repeated | 50.0 (41–58) | 43.2 | 100 | 53.0 | −1.3 (−4 to +2) |
| closed-book | 47.3 (36–56) | 41.7 | 88.9 | 56.1 | −4.0 (−14 to +4) |
| gold evidence | 46.0 (36–55) | 38.6 | 100 | 61.4 (51–75) | −5.3 (−14 to +2) |

Correctness by question type:

| Run | Comparison | Inference | Temporal | Null |
|---|---|---|---|---|
| guesser | 54.0 | 34.4 | 43.2 | 100 |
| baseline | 31.7 | 84.4 | 32.4 | 100 |
| closed-book | 15.9 | 96.9 | 37.8 | 88.9 |
| gold evidence | 28.6 | 96.9 | 5.4 | 100 |

### Finding 1: no row beats the guesser

Every interval overlaps every other. On this subset the baseline pipeline cannot be told apart from always answering with the most common answer for the question's type.

### Finding 2: the gold-evidence row is not a ceiling, and that explains the baseline

Given the gold sentences directly, the generator declined 80 of 100 comparison and temporal questions.

- The questions name their sources and dates: "Does the TechCrunch article…", "between the report published on October 2 and the one on October 7".
- The passages are bare text with no source or date. The generator cannot tell which passage is the TechCrunch one, and the prompt tells it to decline when the passages are not enough. One answer said so: "The passages don't say which articles they come from".
- Inference questions do not depend on attribution, and gold evidence takes them to 96.9%.

So the baseline has two separate problems on comparison and temporal questions: retrieval misses evidence, and the prompt hides the source and date the question asks about. Fixing retrieval alone would not move these answers much. The second problem belongs to Stage 8 (headers) and Stage 9 (abstention).

The row as built does not bound what retrieval can add. A gold-evidence row with source and date headers would. It is not run yet; it fits with Stage 8, which introduces those headers.

### Finding 3: half the answerable questions are declined, mostly after a retrieval miss

Of 132 answerable questions in the baseline run:

| | Abstained | Right | Wrong |
|---|---|---|---|
| Full support in the top 5 (22 questions) | 6 | 15 | 1 |
| Not full support (110 questions) | 63 | 44 | 3 |

- 69 abstentions against 4 wrong answers. The abstention line turns nearly every failure into a refusal, which matches the Stage 1 notes.
- 63 of the 69 follow incomplete retrieval. The other 6 had every gold sentence in the prompt.
- 44 answers were right without full support. A right answer does not show that retrieval worked.

### Finding 4: the generator knows the inference answers

Closed-book, with no passages, it answers 96.9% of inference questions (Sam Bankman-Fried, OpenAI, Google and similar). Answer correctness on inference questions therefore says almost nothing about retrieval. Comparison questions are the opposite: 15.9% closed-book.

### Finding 5: run-to-run noise

Two identical baseline runs, same prompts, same retrieval:

- 97 of 150 answers are the same text.
- 6 of 150 change between right and wrong. The difference in correctness is −1.3 points (−4.3 to +1.8).
- Faithfulness verdicts changed on 9 of 60 answers.

A difference of a few points in answer correctness from single runs on this subset is inside the noise of asking twice.

### Finding 6: what the cluster bootstrap changes

| Metric | Redrawing questions | Redrawing clusters |
|---|---|---|
| recall@5 (441 questions, 120 clusters) | 45.0–51.5 | 41.2–57.2 |
| correct (150 questions, 75 clusters) | 43.3–59.3 | 42.2–59.5 |

For retrieval the honest interval is about 2.5 times wider. A 3-point gain in recall@5 is far inside either unpaired interval, so later stages report paired differences.

### Finding 7: the judge

- **Against the string comparison.** Across the four runs the judge agreed on every answer the comparison had settled, except one in the repeated run. This covers bare answers only.
- **Against a larger judge, correctness.** On 40 answers the comparison could not settle, `gpt-6-luna` and `gpt-6.1-sol` gave the same verdict on all 40.
- **Against a larger judge and two libraries, faithfulness.** On 20 answers from the gold-evidence run, chosen so half were ones `gpt-6-luna` failed:

| Scorer | Result on the same 20 answers |
|---|---|
| `gpt-6-luna`, hand-written prompt | 10 supported |
| `gpt-6.1-sol`, same prompt | 17 supported |
| DeepEval 4.2.8 with `gpt-6-luna` | mean 0.97; 19 answers scored 1.0 |
| RAGAS 0.4.3 with `gpt-6-luna` | mean 0.68; 9 answers scored 1.0 |

The two judges disagree on 9 of 20, and the two libraries disagree with each other. One `gpt-6-luna` verdict failed "Sam Bankman-Fried" because the passages said only "Bankman-Fried".

- **Against a person: deferred.** The label sheets are written (`results/stage02_labels_correctness.csv`, 40 rows; `results/stage02_labels_faithfulness.csv`, 20 rows) and not yet filled in. `python scripts/judge_agreement.py score` reports agreement once they are.

Where that leaves the two judges:

- **Correctness is used as it is.** It is checked against the string comparison and a second judge, and not against a person.
- **Faithfulness is unvalidated.** No faithfulness number in this repo should be read as a result until the 20 labels are done. That has to happen before Stage 8, the first stage that uses faithfulness to make a choice.

### Framework equivalents

- **ranx:** MRR@10, nDCG@10 and recall@5 match the hand-written functions to nine decimal places in `test_hand_written_metrics_match_ranx`.
- **RAGAS and DeepEval:** [framework_equivalents/faithfulness.py](../../framework_equivalents/faithfulness.py). RAGAS 0.4.3 needs `openai<2` and `langchain-community<0.4`, so it runs in its own virtual environment.

### Cost

Stage 2 spent about $3.10: two baseline answer runs at $1.31 each, closed-book $0.14, gold evidence $0.17, and under $0.20 for all judging.

## When you would choose differently in production

- **You would not have gold spans for every question.** Label evidence for a small set and use answer-level metrics with human review for the rest.
- **Use a 1–5 or rubric judge** when answers are long and partly right. Binary fits short factual answers.
- **Validate the judge on your own data before trusting a number,** and again whenever the judge model changes. Hosted models are retired and replaced.
- **Track false abstentions as a product metric.** A system that declines half of what it could answer is safe and of little use; the right balance depends on the cost of a wrong answer.
- **Sample real traffic.** Offline sets drift from what users ask.
- **Report intervals over the unit that is independent** in your data: user, document family or conversation, not single questions.

## Questions you should now be able to answer

1. When do recall@k and MRR disagree?
2. Why is recall@5 unfair across chunk sizes?
3. How do you know the judge is right?
4. Is a 3-point gain on 500 questions real?
5. Why did the gold-evidence row score below the baseline?
6. Why is answer correctness a weak instrument for inference questions here?

<details>
<summary>Short answers</summary>

1. When a question needs several pieces of evidence. MRR looks only at the first relevant chunk; recall and full-support count all of them.
2. A fixed number of chunks is a different amount of text for each chunk size. A fixed token budget gives every chunker the same amount.
3. You do not assume it. You compare its verdicts with human labels on a sample that includes the cases it fails, check it on answers a string comparison already settles, and compare it with a second judge.
4. Not from two separate intervals: the baseline's recall@5 interval is 16 points wide. It can be real if the paired difference on the same questions has an interval that excludes 0, and for answer metrics it also has to exceed the noise of a repeated run.
5. The gold sentences carry no source or date, the questions ask about named sources and dates, and the prompt says to decline when the passages are not enough. The generator declined 80 of 100 comparison and temporal questions.
6. The generator answers 97% of them with no passages at all, so a right answer does not depend on what was retrieved.

</details>
