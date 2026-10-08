from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ragbasics.config import PipelineConfig, load_config
from ragbasics.context.prompts import ABSTAIN_ANSWER, SYSTEM_PROMPT, build_user_prompt
from ragbasics.costs import embedding_cost, generation_cost
from ragbasics.embedding.hashing import HashingEmbedder
from ragbasics.embedding.openai_embedder import OpenAIEmbedder
from ragbasics.generation.llm import AnthropicGenerator
from ragbasics.pipeline import IndexMismatchError, Pipeline
from ragbasics.types import Chunk, Document

REPO_ROOT = Path(__file__).resolve().parents[1]

DOCS = [
    Document("doc_a", "The harbour ferry was delayed by fog on Tuesday.", {"source": "Port News"}),
    Document("doc_b", "Quarterly revenue at the bakery chain rose nine percent."),
    Document("doc_c", "Volcanic ash closed the airport for three days."),
]


def offline_config(tmp_path: Path, **overrides) -> PipelineConfig:
    settings = {
        "name": "test",
        "index_dir": tmp_path / "index",
        "chunker": {"name": "fixed", "params": {"size": 64}},
        "embedder": {"name": "hashing", "params": {"dimensions": 256}},
        "store": {"name": "numpy"},
        "top_k": 2,
        "generator": {"name": "echo"},
    }
    return PipelineConfig.model_validate(settings | overrides)


def test_repo_configs_load_and_build():
    for name in ("baseline", "offline"):
        cfg = load_config(REPO_ROOT / "configs" / f"{name}.yaml", PipelineConfig)
        Pipeline(cfg)  # builds every component; needs no API key
    assert cfg.top_k == 5
    for path in sorted((REPO_ROOT / "configs" / "stage03_chunking").glob("*.yaml")):
        cfg = load_config(path, PipelineConfig)
        Pipeline(cfg)
        assert cfg.name == f"chunk-{path.stem}" and cfg.name.startswith(cfg.index_dir.name)
    for path in sorted((REPO_ROOT / "configs" / "stage04_embedding").glob("*.yaml")):
        cfg = load_config(path, PipelineConfig)
        Pipeline(cfg.model_copy(update={"embedding_cache": None}))  # loads no model
        assert cfg.name == f"embed-{path.stem}" == cfg.index_dir.name
        # One variable changes in Stage 4: the chunker is the best so far.
        assert cfg.chunker.model_dump() == {"name": "recursive", "params": {"size": 128}}


def test_ingest_then_ask_returns_a_full_trace(tmp_path):
    pipeline = Pipeline(offline_config(tmp_path))
    report = pipeline.ingest(DOCS)
    assert (report.documents, report.chunks, report.skipped) == (3, 3, 0)

    trace = pipeline.ask("Why was the ferry delayed?")
    assert [c.rank for c in trace.candidates] == [1, 2]
    assert trace.candidates[0].chunk.doc_id == "doc_a"
    assert trace.candidates[0].score > trace.candidates[1].score
    assert trace.candidates[0].chunk.metadata == {"source": "Port News"}
    assert "harbour ferry" in trace.user_prompt
    assert "harbour ferry" in trace.answer
    assert trace.system_prompt == SYSTEM_PROMPT
    assert set(trace.timings) == {"embed_query", "search", "generate"}
    assert trace.models == {"embedder": "hashing-256", "generator": "echo"}


def test_index_reloads_from_disk_and_gives_the_same_results(tmp_path):
    cfg = offline_config(tmp_path)
    first = Pipeline(cfg)
    first.ingest(DOCS)

    second = Pipeline(cfg)
    assert second.load() is True
    question = "What closed the airport?"
    assert second.ask(question).candidates == first.ask(question).candidates


def test_load_returns_false_when_there_is_no_index(tmp_path):
    assert Pipeline(offline_config(tmp_path)).load() is False


def test_ingesting_the_same_documents_twice_adds_nothing(tmp_path):
    pipeline = Pipeline(offline_config(tmp_path))
    pipeline.ingest(DOCS)
    report = pipeline.ingest(DOCS)
    assert (report.documents, report.skipped, report.chunks) == (0, 3, 0)
    assert len(pipeline.store) == 3


def test_index_built_with_another_embedder_is_refused(tmp_path):
    Pipeline(offline_config(tmp_path)).ingest(DOCS)
    other = offline_config(tmp_path, embedder={"name": "hashing", "params": {"dimensions": 128}})
    with pytest.raises(IndexMismatchError):
        Pipeline(other).load()


def test_asking_an_empty_index_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        Pipeline(offline_config(tmp_path)).ask("anything")


def test_costs_go_to_the_ledger_only_when_money_was_spent(tmp_path):
    ledger = tmp_path / "ledger.csv"
    pipeline = Pipeline(offline_config(tmp_path), ledger=ledger)
    pipeline.ingest(DOCS)
    pipeline.ask("ferry")
    assert not ledger.exists()  # the offline components are free


def test_prompt_numbers_passages_in_rank_order_and_ends_with_the_question():
    chunks = [Chunk("d:0", "d", "First passage.\n", 0, 15), Chunk("d:1", "d", "Second.", 15, 22)]
    prompt = build_user_prompt("Who?", chunks)
    assert prompt == (
        '<passages>\n<passage id="1">\nFirst passage.\n</passage>\n\n'
        '<passage id="2">\nSecond.\n</passage>\n</passages>\n\nQuestion: Who?'
    )
    assert ABSTAIN_ANSWER in SYSTEM_PROMPT


def test_hashing_embedder_scores_shared_words_higher():
    vectors = HashingEmbedder(256).embed(["red apple", "red apple pie", "blue whale"]).vectors
    assert vectors[0] @ vectors[1] > vectors[0] @ vectors[2]


def test_cost_arithmetic():
    assert embedding_cost("text-embedding-3-small", 1_000_000) == pytest.approx(0.02)
    assert generation_cost("claude-sonnet-5-5", 1_000_000, 100_000) == pytest.approx(3.0)
    assert generation_cost("echo", 1_000, 1_000) == 0.0


# --- API adapters, exercised against stub clients (no network) -----------------------------


class StubOpenAI:
    """Returns vector [i, 1] for the i-th text overall, with the batch reversed, so the
    test also checks that results are put back in input order."""

    def __init__(self):
        self.calls: list[list[str]] = []
        self.embeddings = self
        self.seen = 0

    def create(self, model, input):
        self.calls.append(input)
        data = [
            SimpleNamespace(index=i, embedding=[float(self.seen + i), 1.0])
            for i in range(len(input))
        ]
        self.seen += len(input)
        usage = SimpleNamespace(total_tokens=len(input) * 3)
        return SimpleNamespace(data=data[::-1], usage=usage)


def test_openai_embedder_batches_keeps_order_and_counts_tokens():
    client = StubOpenAI()
    embedded = OpenAIEmbedder(batch_size=2, client=client).embed(["a", "b", "c"])
    assert client.calls == [["a", "b"], ["c"]]
    assert embedded.vectors.dtype == np.float32
    assert embedded.vectors.tolist() == [[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]]
    assert embedded.tokens == 9


class StubAnthropic:
    def __init__(self):
        self.messages = self
        self.request = None

    def create(self, **request):
        self.request = request
        return SimpleNamespace(
            content=[
                SimpleNamespace(type="thinking", thinking=""),
                SimpleNamespace(type="text", text=" Sam Bankman-Fried \n"),
            ],
            usage=SimpleNamespace(input_tokens=2700, output_tokens=40),
            stop_reason="end_turn",
        )


def test_anthropic_generator_sends_the_prompt_and_reads_only_text_blocks():
    client = StubAnthropic()
    generation = AnthropicGenerator(client=client).generate("system text", "user text")
    assert generation.text == "Sam Bankman-Fried"
    assert (generation.input_tokens, generation.output_tokens) == (2700, 40)
    assert generation.stop_reason == "end_turn"
    assert client.request["model"] == "claude-sonnet-5-5"
    assert client.request["system"] == "system text"
    assert client.request["messages"] == [{"role": "user", "content": "user text"}]
    assert "temperature" not in client.request


def test_parent_child_retrieval_returns_each_parent_once(tmp_path):
    text = ("The harbour ferry was delayed by fog. The ferry left at noon. " * 4
            + "\n\nQuarterly revenue at the bakery rose. The bakery sold more bread. " * 4)
    cfg = offline_config(
        tmp_path,
        chunker={"name": "parent_child", "params": {"parent_size": 64, "child_size": 16}},
        overfetch=4,
    )
    pipeline = Pipeline(cfg)
    pipeline.ingest([Document("doc_p", text)])
    parents = {(c.start_char, c.end_char) for c in pipeline.store.chunks}
    assert len(pipeline.store) > len(parents) > 1  # several children per parent

    trace = pipeline.retrieve("Why was the harbour ferry delayed?", k=len(parents))
    spans = [(c.chunk.start_char, c.chunk.end_char) for c in trace.candidates]
    assert sorted(spans) == sorted(parents)  # every parent, none twice
    assert [c.rank for c in trace.candidates] == list(range(1, len(parents) + 1))
    top = trace.candidates[0].chunk
    assert "ferry" in top.text and top.text == text[top.start_char : top.end_char]
