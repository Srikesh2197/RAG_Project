"""Stage 0 experiment: profile the corpus and the questions.

Writes results/stage00_data_profile.md. Run after `make data`.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import tiktoken

from ragbasics.config import DataConfig, load_config
from ragbasics.eval.dataset import largest_cluster_share, read_documents, read_questions
from ragbasics.types import Question

SPLITS = ("dev", "test", "pool", "test_pool")
OUTPUT = Path("results/stage00_data_profile.md")


def percentile(values: list[int], p: float) -> int:
    """Nearest-rank percentile; p in [0, 100]."""
    ordered = sorted(values)
    rank = max(1, round(p / 100 * len(ordered)))
    return ordered[rank - 1]


def table(header: list[str], rows: list[list[object]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(lines)


def spread(values: list[int]) -> list[int]:
    return [min(values), percentile(values, 50), percentile(values, 90), max(values)]


def answer_form(question: Question) -> str:
    answer = question.answer.strip().lower().rstrip(".")
    if answer in ("yes", "no"):
        return "yes / no"
    if answer == "insufficient information":
        return "insufficient information"
    return "entity or phrase"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config, DataConfig)
    processed = cfg.processed_dir

    documents = read_documents(processed / "documents.jsonl")
    by_split = {s: read_questions(processed / f"questions_{s}.jsonl") for s in SPLITS}
    questions = [q for split in by_split.values() for q in split]
    manifest = json.loads((processed / "manifest.json").read_text())
    doc_by_id = {doc.doc_id: doc for doc in documents}
    # text-embedding-3-* models tokenise with cl100k_base, so chunk sizes will be in these units.
    encoding = tiktoken.get_encoding("cl100k_base")

    sections: list[str] = ["# Stage 0 data profile", ""]
    sections.append(f"Dataset `{cfg.repo_id}` at revision `{cfg.revision[:12]}`. Seed {cfg.seed}.")

    # --- Documents ---
    doc_chars = [len(doc.text) for doc in documents]
    doc_tokens = [len(encoding.encode(doc.text, disallowed_special=())) for doc in documents]
    sources = Counter(doc.metadata["source"] for doc in documents)
    categories = Counter(doc.metadata["category"] for doc in documents)
    dates = sorted(doc.metadata["published_at"] for doc in documents)
    sections += [
        "",
        "## Documents",
        "",
        f"- {len(documents)} articles, {sum(doc_tokens):,} tokens in total.",
        f"- {len(sources)} sources. Largest: "
        + ", ".join(f"{name} ({count})" for name, count in sources.most_common(5))
        + ".",
        "- Categories: "
        + ", ".join(f"{name} ({count})" for name, count in categories.most_common())
        + ".",
        f"- Published between {dates[0][:10]} and {dates[-1][:10]}.",
        "",
        table(
            ["Length per article", "min", "median", "90th percentile", "max"],
            [["characters", *spread(doc_chars)], ["tokens", *spread(doc_tokens)]],
        ),
    ]

    # --- Questions ---
    types = sorted({q.question_type for q in questions})
    rows = [
        [t, *[sum(q.question_type == t for q in by_split[s]) for s in SPLITS]] for t in types
    ]
    rows.append(["total", *[len(by_split[s]) for s in SPLITS]])
    subset = Counter(q.question_type for q in by_split["dev"] if q.answer_subset)
    sections += [
        "",
        "## Questions by split and type",
        "",
        table(["type", *SPLITS], rows),
        "",
        f"Answer-quality subset inside dev: {sum(subset.values())} questions ("
        + ", ".join(f"{t} {subset[t]}" for t in types)
        + ").",
    ]

    # --- Answers ---
    forms = Counter(answer_form(q) for q in questions)
    top_answers = Counter(q.answer.strip() for q in questions).most_common(6)
    guess_rows = []
    for split in ("dev", "test"):
        row: list[object] = [split]
        correct = 0
        for t in types:
            answers = Counter(
                q.answer.strip().lower() for q in by_split[split] if q.question_type == t
            )
            answer, count = answers.most_common(1)[0]
            correct += count
            row.append(f"{answer} ({count / sum(answers.values()):.0%})")
        row.append(f"{correct / len(by_split[split]):.1%}")
        guess_rows.append(row)
    sections += [
        "",
        "## Reference answers",
        "",
        "All 2,556 questions:",
        "",
        table(
            ["form", "questions", "share"],
            [[f, n, f"{n / len(questions):.1%}"] for f, n in forms.most_common()],
        ),
        "",
        "Most common answers: "
        + ", ".join(f"{answer} ({n / len(questions):.1%})" for answer, n in top_answers)
        + ".",
        "",
        "Most common answer per question type, and the score from always giving it "
        "(a guesser that reads only the question's type):",
        "",
        table(["split", *types, "guesser's score"], guess_rows),
    ]

    # --- Evidence ---
    answerable = [q for q in questions if q.evidence]
    per_question = Counter(len(q.evidence) for q in answerable)
    distinct_docs = Counter(len({span.doc_id for span in q.evidence}) for q in answerable)
    spans = [span for q in answerable for span in q.evidence]
    span_tokens = [
        len(encoding.encode(doc_by_id[s.doc_id].text[s.start_char : s.end_char])) for s in spans
    ]
    fifths = Counter(
        min(4, int(5 * s.start_char / len(doc_by_id[s.doc_id].text))) for s in spans
    )
    unique_spans = set(spans)
    gold_chars = sum(s.end_char - s.start_char for s in unique_spans)
    mapping = manifest["mapping"]
    sections += [
        "",
        "## Gold evidence",
        "",
        f"- Mapped to character spans: {mapping['mapped']:,} of {mapping['evidence_total']:,} "
        f"({mapping['rate']:.2%}), {mapping['matched_exact']:,} verbatim.",
        f"- Evidence sentences that occur more than once in their article: "
        f"{mapping['sentence_repeats_in_article']} (the span marks the first occurrence).",
        f"- Questions with no evidence (unanswerable): {mapping['questions_without_evidence']}.",
        f"- Articles holding at least one gold span: "
        f"{len({s.doc_id for s in unique_spans})} of {len(documents)}.",
        f"- Gold spans cover {gold_chars / sum(doc_chars):.1%} of all corpus characters "
        f"({len(unique_spans):,} distinct spans).",
        f"- Evidence sentence length in tokens: median {percentile(span_tokens, 50)}, "
        f"90th percentile {percentile(span_tokens, 90)}, max {max(span_tokens)}.",
        "",
        table(
            ["evidence items per answerable question", "questions", "share"],
            [[k, n, f"{n / len(answerable):.1%}"] for k, n in sorted(per_question.items())],
        ),
        "",
        table(
            ["distinct articles per answerable question", "questions", "share"],
            [[k, n, f"{n / len(answerable):.1%}"] for k, n in sorted(distinct_docs.items())],
        ),
        "",
        table(
            ["where in the article the evidence starts", "share of evidence"],
            [
                [f"{20 * i}–{20 * (i + 1)}% of the way through", f"{fifths[i] / len(spans):.1%}"]
                for i in range(5)
            ],
        ),
    ]

    # --- Evidence shared between questions ---
    uses = Counter(span for q in answerable for span in q.evidence)
    cluster_sizes = Counter(q.cluster for q in answerable)
    hub_rows = []
    for span, count in uses.most_common(3):
        doc = doc_by_id[span.doc_id]
        sentence = doc.text[span.start_char : span.end_char]
        hub_rows.append([count, doc.metadata["source"], sentence[:90] + "…"])
    split_rows = []
    for split in ("dev", "test"):
        answerable_in_split = [q for q in by_split[split] if q.evidence]
        split_rows.append(
            [
                split,
                len(answerable_in_split),
                len({q.cluster for q in answerable_in_split}),
                f"{largest_cluster_share(by_split[split]):.1%}",
            ]
        )
    check = manifest["split_check"]
    sections += [
        "",
        "## Evidence shared between questions",
        "",
        f"- {mapping['evidence_total']:,} evidence items are {len(uses):,} distinct sentences. "
        f"A sentence is gold for {percentile(list(uses.values()), 50)} questions at the median "
        f"and {max(uses.values())} at most; {sum(n == 1 for n in uses.values())} are used once.",
        f"- Linking questions that share a sentence gives {len(cluster_sizes)} clusters. "
        f"The largest hold {', '.join(str(n) for _, n in cluster_sizes.most_common(4))} questions; "
        f"{sum(n == 1 for n in cluster_sizes.values())} clusters are a single question.",
        f"- Gold sentences shared by dev and test: {check['gold_spans_shared_by_dev_and_test']}. "
        f"Shared by pool and test: {check['gold_spans_shared_by_pool_and_test']}.",
        "",
        table(["questions using it", "source", "most reused gold sentences"], hub_rows),
        "",
        table(
            ["split", "answerable questions", "clusters represented", "largest cluster's share"],
            split_rows,
        ),
    ]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(sections) + "\n")
    print("\n".join(sections))
    print(f"\nwrote {OUTPUT}")


if __name__ == "__main__":
    main()
