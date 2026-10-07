"""Library equivalents of the hand-written faithfulness judge: RAGAS and DeepEval.

Both libraries score faithfulness the same way: split the answer into claims, check each
claim against the passages, and report the share of claims that are supported. The
hand-written judge asks one yes-or-no question about the whole answer.

Run it in a separate virtual environment, not the project's: RAGAS 0.4.3 needs
openai<2 and langchain-community<0.4, which would downgrade the project's packages.

    python -m venv .venv-ragas && .venv-ragas/bin/pip install "ragas==0.4.3" \
        "langchain-community<0.4" python-dotenv
    .venv-ragas/bin/python framework_equivalents/faithfulness.py ragas

    python -m venv .venv-deepeval && .venv-deepeval/bin/pip install deepeval python-dotenv
    .venv-deepeval/bin/python framework_equivalents/faithfulness.py deepeval

Input: results/stage02_labels_faithfulness.csv (20 answers from the gold-evidence run).
Output: results/stage02_framework_faithfulness.json, one score per item and library.
"""

import csv
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

SHEET = Path("results/stage02_labels_faithfulness.csv")
OUTPUT = Path("results/stage02_framework_faithfulness.json")
MODEL = "gpt-6-luna"


def load_items() -> list[dict]:
    with open(SHEET, newline="") as f:
        items = list(csv.DictReader(f))
    for item in items:
        # The sheet stores passages as "[1] text" lines.
        item["contexts"] = [line.split("] ", 1)[1] for line in item["passages"].split("\n")]
    return items


def score_ragas(items: list[dict]) -> dict[str, float | None]:
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory
    from ragas.metrics.collections import Faithfulness

    metric = Faithfulness(llm=llm_factory(MODEL, client=AsyncOpenAI()))
    scores = {}
    for item in items:
        result = metric.score(
            user_input=item["question"],
            response=item["answer"],
            retrieved_contexts=item["contexts"],
        )
        scores[item["id"]] = float(result.value)
    return scores


def score_deepeval(items: list[dict]) -> dict[str, float | None]:
    from deepeval.metrics import FaithfulnessMetric
    from deepeval.test_case import LLMTestCase

    scores = {}
    for item in items:
        metric = FaithfulnessMetric(model=MODEL, async_mode=False, include_reason=False)
        metric.measure(
            LLMTestCase(
                input=item["question"],
                actual_output=item["answer"],
                retrieval_context=item["contexts"],
            )
        )
        scores[item["id"]] = float(metric.score)
    return scores


def main() -> None:
    load_dotenv()
    os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
    os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")
    library = sys.argv[1] if len(sys.argv) > 1 else ""
    scorers = {"ragas": score_ragas, "deepeval": score_deepeval}
    if library not in scorers:
        raise SystemExit("usage: faithfulness.py ragas|deepeval")
    scores = scorers[library](load_items())
    results = json.loads(OUTPUT.read_text()) if OUTPUT.exists() else {}
    results[library] = scores
    OUTPUT.write_text(json.dumps(results, indent=2) + "\n")
    print(f"{library}: mean faithfulness {sum(scores.values()) / len(scores):.2f} "
          f"over {len(scores)} answers. Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
