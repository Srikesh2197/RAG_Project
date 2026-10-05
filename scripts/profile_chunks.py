"""Profile what a config's chunker does to the corpus. Needs no API key.

    python scripts/profile_chunks.py --config configs/baseline.yaml

Reports chunk counts and sizes, and how many gold evidence sentences of the development
questions are cut in two by a chunk boundary.
"""

import argparse
import statistics
from collections import defaultdict
from pathlib import Path

import tiktoken

from ragbasics.config import PipelineConfig, load_config
from ragbasics.eval.dataset import read_documents, read_questions
from ragbasics.pipeline import Pipeline

PROCESSED = Path("data/processed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    args = parser.parse_args()

    pipeline = Pipeline(load_config(args.config, PipelineConfig))
    documents = read_documents(PROCESSED / "documents.jsonl")
    chunks = pipeline.chunk(documents)
    encoding = tiktoken.get_encoding("cl100k_base")
    sizes = [len(encoding.encode(c.text, disallowed_special=())) for c in chunks]
    per_doc = defaultdict(int)
    for chunk in chunks:
        per_doc[chunk.doc_id] += 1

    print(f"chunks: {len(chunks)} from {len(documents)} documents")
    print(f"chunks per document: median {statistics.median(per_doc.values())}, "
          f"max {max(per_doc.values())}")
    full = sum(size == max(sizes) for size in sizes)
    print(f"chunk tokens: max {max(sizes)}, median {statistics.median(sizes)}, "
          f"{full} at the maximum, {sum(size < 100 for size in sizes)} under 100")

    starts = defaultdict(set)
    for chunk in chunks:
        starts[chunk.doc_id].add(chunk.start_char)
    spans = {s for q in read_questions(PROCESSED / "questions_dev.jsonl") for s in q.evidence}
    cut = sum(
        any(s.start_char < boundary < s.end_char for boundary in starts[s.doc_id]) for s in spans
    )
    print(f"development gold sentences: {len(spans)}, cut by a chunk boundary: {cut} "
          f"({cut / len(spans):.1%})")


if __name__ == "__main__":
    main()
