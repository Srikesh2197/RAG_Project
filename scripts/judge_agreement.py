"""Stage 2 experiment: how often does the judge agree with a person?

    python scripts/judge_agreement.py sample   # write the two label sheets
    python scripts/judge_agreement.py score    # after the sheets are filled in

`sample` picks answers the judge had to decide (the string comparison could not) and
writes them, shuffled and without any verdict, to two CSV files:

    results/stage02_labels_correctness.csv    40 answers: is the answer correct?
    results/stage02_labels_faithfulness.csv   20 answers: do the passages support it?

It also asks a second, larger judge model for its verdict on the same items, so the two
judges can be compared against the same human labels. All verdicts go to
results/stage02_judge_verdicts.json, which `score` reads.

Fill the `human` column with y or n, then run `score`.
"""

import argparse
import csv
import json
import random
from pathlib import Path

from dotenv import load_dotenv

from ragbasics.cache import CachedGenerator, DiskCache
from ragbasics.config import EvalConfig, load_config
from ragbasics.costs import append_ledger, generation_cost
from ragbasics.eval.dataset import read_documents, read_questions
from ragbasics.eval.judge import judge_correctness, judge_faithfulness
from ragbasics.eval.runner import gold_context
from ragbasics.eval.stats import cohen_kappa
from ragbasics.generation.llm import OpenAIGenerator

RUNS = Path("runs")
CORRECTNESS_RUNS = ("baseline", "baseline-closed_book", "baseline-gold")
FAITHFULNESS_RUN = "baseline-gold"  # its passages are 2 to 4 sentences, quick to read
CORRECTNESS_SHEET = Path("results/stage02_labels_correctness.csv")
FAITHFULNESS_SHEET = Path("results/stage02_labels_faithfulness.csv")
VERDICTS = Path("results/stage02_judge_verdicts.json")
REPORT = Path("results/stage02_judge_agreement.md")
SECOND_JUDGE = "gpt-6.1-sol"
SEED = 7


def rows_of(run: str) -> list[dict]:
    with open(RUNS / run / "questions.jsonl") as f:
        return [json.loads(line) | {"run": run} for line in f]


def pick(pool: list[dict], verdict_key: str, n: int, rng: random.Random) -> list[dict]:
    """Up to n/2 items the judge failed and the rest from those it passed.

    Failures are rare, so a plain random sample would hold too few of them to say
    anything about how often the judge wrongly fails or wrongly passes an answer.
    """
    failed = [row for row in pool if row[verdict_key] is False]
    passed = [row for row in pool if row[verdict_key] is True]
    rng.shuffle(failed)
    rng.shuffle(passed)
    chosen = failed[: n // 2]
    chosen += passed[: n - len(chosen)]
    rng.shuffle(chosen)
    return chosen


def cmd_sample(args: argparse.Namespace) -> None:
    for sheet in (CORRECTNESS_SHEET, FAITHFULNESS_SHEET):
        if sheet.exists() and not args.force:
            raise SystemExit(f"{sheet} exists and may hold labels. Use --force to replace it.")
    cfg = load_config(args.eval_config, EvalConfig)
    rng = random.Random(SEED)

    # Correctness: answers the string comparison could not settle, one per distinct answer.
    seen: set[tuple[str, str]] = set()
    pool = []
    for run in CORRECTNESS_RUNS:
        for row in rows_of(run):
            key = (row["question_id"], row.get("answer", ""))
            if row.get("match", 0) is None and row.get("judge_correct") is not None:
                if key not in seen:
                    seen.add(key)
                    pool.append(row)
    correctness = pick(pool, "judge_correct", 40, rng)

    pool = [r for r in rows_of(FAITHFULNESS_RUN) if r.get("judge_faithful") is not None]
    faithfulness = pick(pool, "judge_faithful", 20, rng)
    questions = {q.question_id: q for q in read_questions(cfg.questions)}
    text_by_doc = {d.doc_id: d.text for d in read_documents(cfg.documents)}
    contexts = {
        row["question_id"]: gold_context(questions[row["question_id"]], text_by_doc)
        for row in faithfulness
    }

    # The second judge, on the same items.
    second = CachedGenerator(OpenAIGenerator(SECOND_JUDGE), DiskCache(cfg.cache))
    tokens_in = tokens_out = 0
    verdicts: dict[str, dict] = {}
    for i, row in enumerate(correctness, start=1):
        v = judge_correctness(second, row["question"], row["reference"], row["answer"])
        if not v.generation.cached:
            tokens_in += v.generation.input_tokens
            tokens_out += v.generation.output_tokens
        verdicts[f"c{i:02d}"] = {
            "run": row["run"],
            "question_id": row["question_id"],
            cfg.judge.params["model"]: row["judge_correct"],
            SECOND_JUDGE: v.passed,
        }
    for i, row in enumerate(faithfulness, start=1):
        context = contexts[row["question_id"]]
        v = judge_faithfulness(second, row["question"], row["answer"], context)
        if not v.generation.cached:
            tokens_in += v.generation.input_tokens
            tokens_out += v.generation.output_tokens
        verdicts[f"f{i:02d}"] = {
            "run": row["run"],
            "question_id": row["question_id"],
            cfg.judge.params["model"]: row["judge_faithful"],
            SECOND_JUDGE: v.passed,
        }
    cost = generation_cost(SECOND_JUDGE, tokens_in, tokens_out)
    if cost:
        append_ledger(
            RUNS / "cost_ledger.csv", "judge_agreement", SECOND_JUDGE, tokens_in, tokens_out, cost
        )

    CORRECTNESS_SHEET.parent.mkdir(parents=True, exist_ok=True)
    with open(CORRECTNESS_SHEET, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "question", "reference_answer", "answer", "human"])
        for i, row in enumerate(correctness, start=1):
            writer.writerow([f"c{i:02d}", row["question"], row["reference"], row["answer"], ""])
    with open(FAITHFULNESS_SHEET, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "passages", "question", "answer", "human"])
        for i, row in enumerate(faithfulness, start=1):
            passages = "\n".join(
                f"[{n}] {c.text}" for n, c in enumerate(contexts[row["question_id"]], start=1)
            )
            writer.writerow([f"f{i:02d}", passages, row["question"], row["answer"], ""])
    VERDICTS.write_text(json.dumps(verdicts, indent=2) + "\n")
    print(
        f"Wrote {CORRECTNESS_SHEET} ({len(correctness)} rows) and {FAITHFULNESS_SHEET} "
        f"({len(faithfulness)} rows). {SECOND_JUDGE} cost ${cost:.2f}."
    )


def read_labels(sheet: Path) -> dict[str, bool]:
    labels = {}
    with open(sheet, newline="") as f:
        for row in csv.DictReader(f):
            value = row["human"].strip().lower()
            if value in ("y", "yes", "1", "true"):
                labels[row["id"]] = True
            elif value in ("n", "no", "0", "false"):
                labels[row["id"]] = False
    return labels


def cmd_score(args: argparse.Namespace) -> None:
    verdicts = json.loads(VERDICTS.read_text())
    lines = [
        "# Stage 2: judge agreement with human labels",
        "",
        "Items were chosen so that about half are ones the first judge failed, so these "
        "rates describe the judge on hard cases, not on a typical run.",
    ]
    for title, sheet, positive in (
        ("Correctness", CORRECTNESS_SHEET, "correct"),
        ("Faithfulness (gold-evidence run, short passages)", FAITHFULNESS_SHEET, "supported"),
    ):
        human = read_labels(sheet)
        if not human:
            raise SystemExit(f"No labels in {sheet}. Fill the `human` column with y or n.")
        judges = [k for k in next(iter(verdicts.values())) if k not in ("run", "question_id")]
        lines += [
            "",
            f"## {title}: {len(human)} labelled",
            "",
            f"| judge | agrees with human | kappa | said {positive}, human said no | "
            f"said not {positive}, human said yes |",
            "|---|---|---|---|---|",
        ]
        for judge in judges:
            ids = [i for i in human if verdicts[i][judge] is not None]
            h = [human[i] for i in ids]
            j = [verdicts[i][judge] for i in ids]
            agree = sum(x == y for x, y in zip(h, j, strict=True))
            too_kind = sum(y and not x for x, y in zip(h, j, strict=True))
            too_harsh = sum(x and not y for x, y in zip(h, j, strict=True))
            lines.append(
                f"| {judge} | {agree} of {len(ids)} ({agree / len(ids):.0%}) | "
                f"{cohen_kappa(h, j):.2f} | {too_kind} | {too_harsh} |"
            )
        disagreements = [
            i for i in human if any(verdicts[i][j] not in (None, human[i]) for j in judges)
        ]
        listed = ", ".join(disagreements) or "none"
        lines += ["", f"Items where a judge and the human differ: {listed}."]
    REPORT.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {REPORT}")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    sample = commands.add_parser("sample")
    sample.add_argument("--eval-config", default="configs/eval.yaml")
    sample.add_argument("--force", action="store_true")
    sample.set_defaults(run=cmd_sample)
    commands.add_parser("score").set_defaults(run=cmd_score)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
