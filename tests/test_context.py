"""Context assembly (Stage 8): which chunks reach the prompt, merged, labelled and ordered
how. No API keys or downloads."""

from pathlib import Path

import numpy as np
import pytest

from ragbasics.config import PipelineConfig, load_config
from ragbasics.context.assemble import (
    Assembler,
    dedupe,
    merge_neighbours,
    order_chronological,
    order_edges,
    prompt_header,
)
from ragbasics.context.mmr import min_max, mmr
from ragbasics.context.prompts import (
    HEADER_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_prompt,
    build_user_prompt,
    render_passages,
)
from ragbasics.eval.judge import faithfulness_prompt
from ragbasics.pipeline import Pipeline
from ragbasics.types import Candidate, Chunk, Document, Passage

REPO_ROOT = Path(__file__).resolve().parents[1]

META = {
    "a": {"title": "Ferry delayed", "source": "Port News",
          "published_at": "2023-10-07T08:00:00+00:00"},
    "b": {"title": "Bakery results", "source": "TechCrunch",
          "published_at": "2023-10-02T08:00:00+00:00"},
    "c": {"title": "Airport closed"},  # no source, no date
}
TEXT = {
    "a": "The ferry left late. Fog covered the harbour. It returned after dark.",
    "b": "Revenue rose nine percent. Shares followed.",
    "c": "Ash closed the airport.",
}


def chunk(doc: str, start: int, end: int) -> Chunk:
    return Chunk(f"{doc}:{start}", doc, TEXT[doc][start:end], start, end, metadata=META[doc])


def ranked(*chunks: Chunk, scores: list[float] | None = None) -> list[Candidate]:
    scores = scores or [1.0 - i / 10 for i in range(len(chunks))]
    pairs = zip(chunks, scores, strict=True)
    return [Candidate(c, s, rank) for rank, (c, s) in enumerate(pairs, start=1)]


A1, A2, A3 = chunk("a", 0, 21), chunk("a", 21, 45), chunk("a", 45, 70)
B1, B2 = chunk("b", 0, 27), chunk("b", 27, 43)
C1 = chunk("c", 0, 23)


# --- The default changes nothing ------------------------------------------------------


def stage1_prompt(question: str, chunks: list[Chunk]) -> str:
    """The prompt as Stage 1 wrote it, kept here verbatim to pin the format."""
    passages = "\n\n".join(
        f'<passage id="{i}">\n{c.text.strip()}\n</passage>' for i, c in enumerate(chunks, 1)
    )
    return f"<passages>\n{passages}\n</passages>\n\nQuestion: {question}"


def test_default_assembly_gives_the_stage1_prompt_byte_for_byte():
    candidates = ranked(B1, A2, C1, A1)
    assembly = Assembler(top_k=3).assemble(candidates)
    assert assembly.chunks == [B1, A2, C1]
    assert build_user_prompt("Why?", assembly.passages) == stage1_prompt("Why?", [B1, A2, C1])
    assert build_prompt("Why?", assembly.passages)[0] == SYSTEM_PROMPT


def test_plain_chunks_and_passages_render_the_same():
    assert render_passages([A1, B1]) == render_passages([Passage((A1,), 1), Passage((B1,), 2)])


def test_the_judge_sees_the_passages_the_generator_saw():
    labelled = Assembler(top_k=2, header=True).assemble(ranked(A2, B1)).passages
    prompt = faithfulness_prompt("Why?", "Fog.", labelled)
    assert prompt.startswith(render_passages(labelled))
    assert "Port News, October 7, 2023." in prompt
    # Without headers the judge prompt is the one Stage 2 cached.
    old = stage1_prompt("Why?", [A2, B1]).replace("\n\nQuestion: Why?", "")
    expected = f"{old}\n\nQuestion: Why?\n\nAnswer: Fog."
    assert faithfulness_prompt("Why?", "Fog.", [A2, B1]) == expected


# --- Headers ----------------------------------------------------------------------------


def test_header_names_title_source_and_written_out_date():
    passages = Assembler(top_k=3, header=True).assemble(ranked(A2, B1, C1)).passages
    assert [p.header for p in passages] == [
        "Ferry delayed. Port News, October 7, 2023.",
        "Bakery results. TechCrunch, October 2, 2023.",
        "Airport closed.",  # what the chunk has, nothing invented
    ]
    system, user = build_prompt("Why?", passages)
    assert system == HEADER_SYSTEM_PROMPT != SYSTEM_PROMPT
    assert '<passage id="1">\nFerry delayed. Port News, October 7, 2023.\nFog covered' in user


def test_a_title_ending_in_a_question_mark_gets_no_extra_full_stop():
    meta = {"title": "AI Alliance, but to what end?", "source": "TechCrunch",
            "published_at": "2023-12-05T10:00:00+00:00"}
    assert prompt_header(meta) == "AI Alliance, but to what end? TechCrunch, December 5, 2023."
    assert prompt_header({"title": "Done."}) == "Done."


def test_a_chunk_with_no_metadata_gets_no_header_line():
    bare = Chunk("x:0", "x", "Some text.", 0, 10)
    passages = Assembler(top_k=1, header=True).assemble(ranked(bare)).passages
    assert passages[0].header == ""
    assert build_prompt("Q?", passages)[0] == SYSTEM_PROMPT


def test_wrap_labels_gold_chunks_in_the_order_given():
    passages = Assembler(top_k=1, header=True).wrap([B1, A1, C1])  # top_k does not cut
    assert [p.chunks[0] for p in passages] == [B1, A1, C1]
    assert passages[1].header.startswith("Ferry delayed.")


# --- Deduplication ----------------------------------------------------------------------


def test_dedupe_drops_repeated_text_and_frees_the_slot():
    copy = Chunk("d:0", "c", "  ash CLOSED the\nairport. ", 0, 26, metadata=META["c"])
    kept, dropped = dedupe(ranked(C1, copy, A1))
    assert dropped == 1 and [c.chunk for c in kept] == [C1, A1]
    assembly = Assembler(top_k=2, pool=3, dedupe=True).assemble(ranked(C1, copy, A1))
    assert assembly.chunks == [C1, A1] and assembly.notes["duplicates"] == 1


def test_without_a_larger_pool_dedupe_cannot_reach_below_top_k():
    copy = Chunk("d:0", "c", TEXT["c"], 0, 23)
    assembly = Assembler(top_k=2, dedupe=True).assemble(ranked(C1, copy, A1))
    assert assembly.chunks == [C1]


# --- Merging neighbours -----------------------------------------------------------------


def test_touching_chunks_of_one_article_merge_back_into_its_text():
    passages = merge_neighbours(ranked(A3, B1, A2))
    assert [p.chunks for p in passages] == [(A2, A3), (B1,)]
    assert passages[0].text == TEXT["a"][21:70]
    assert passages[0].rank == 1  # the best rank among its chunks


def test_chunks_with_a_gap_between_them_stay_apart():
    passages = merge_neighbours(ranked(A1, A3))
    assert [p.chunks for p in passages] == [(A1,), (A3,)]


def test_overlapping_chunks_merge_without_repeating_text():
    overlap = chunk("a", 15, 45)
    passages = merge_neighbours(ranked(A1, overlap))
    assert passages[0].text == TEXT["a"][0:45]


def test_merge_in_the_assembler_reports_how_many_chunks_joined():
    assembly = Assembler(top_k=4, merge=True, header=True).assemble(ranked(A2, B1, A1, A3))
    assert assembly.notes["merged"] == 2 and assembly.notes["passages"] == 2
    titles = [p.header.split(".")[0] for p in assembly.passages]
    assert titles == ["Ferry delayed", "Bakery results"]
    assert assembly.chunks == [A1, A2, A3, B1]


# --- Ordering ---------------------------------------------------------------------------


def passages_ranked(n: int) -> list[Passage]:
    return [Passage((chunk("a", i, i + 1),), rank=i + 1) for i in range(n)]


def test_edges_puts_the_two_best_at_the_ends_and_the_weakest_in_the_middle():
    assert [p.rank for p in order_edges(passages_ranked(5))] == [1, 3, 5, 4, 2]
    # With an even count the best lands last, next to the question (as in LangChain).
    assert [p.rank for p in order_edges(passages_ranked(4))] == [2, 4, 3, 1]


def test_chronological_reads_articles_oldest_first_and_undated_last():
    passages = [Passage((c,), r) for r, c in enumerate([C1, A3, B1, A1], start=1)]
    order = [p.chunks[0] for p in order_chronological(passages)]
    assert order == [B1, A1, A3, C1]  # TechCrunch Oct 2, Port News Oct 7 in text order, undated


def test_unknown_order_is_refused():
    with pytest.raises(ValueError, match="unknown order"):
        Assembler(top_k=2, order="random")


# --- Token budget -----------------------------------------------------------------------


def test_token_budget_drops_whole_passages_from_the_lowest_rank():
    full = Assembler(top_k=3).assemble(ranked(A1, B1, C1))
    budget = Assembler(top_k=3, token_budget=12).assemble(ranked(A1, B1, C1))
    assert len(budget.passages) < len(full.passages)
    assert budget.chunks == full.chunks[: len(budget.chunks)]  # nothing cut mid-passage
    assert budget.notes["over_budget"] == len(full.passages) - len(budget.passages)


def test_token_budget_keeps_at_least_one_passage():
    assembly = Assembler(top_k=2, token_budget=1).assemble(ranked(A1, B1))
    assert assembly.chunks == [A1]


# --- Maximal marginal relevance ---------------------------------------------------------

# Items 0 and 1 say nearly the same thing; 2 and 3 are different from everything.
RELEVANCE = np.array([1.0, 0.9, 0.8, 0.1])
SIMILARITY = np.array(
    [
        [1.0, 0.95, 0.1, 0.0],
        [0.95, 1.0, 0.1, 0.0],
        [0.1, 0.1, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def test_mmr_by_hand():
    # lam 0.5, after picking 0: item 1 scores .45 - .475 = -.025, item 2 .40 - .05 = .35,
    # item 3 .05 - 0 = .05. Then item 3 (.05) beats item 1 (still -.025).
    assert mmr(RELEVANCE, SIMILARITY, 3, 0.5) == [0, 2, 3]


def test_mmr_at_lambda_one_is_relevance_order_and_at_zero_takes_the_most_different():
    assert mmr(RELEVANCE, SIMILARITY, 4, 1.0) == [0, 1, 2, 3]
    assert mmr(RELEVANCE, SIMILARITY, 3, 0.0) == [0, 3, 2]


def test_mmr_edge_cases():
    assert mmr(np.array([]), np.zeros((0, 0)), 3, 0.5) == []
    assert mmr(RELEVANCE, SIMILARITY, 10, 0.5) == [0, 2, 3, 1]  # k above n returns all
    with pytest.raises(ValueError):
        mmr(RELEVANCE, SIMILARITY, 2, 1.5)


def test_min_max_scales_reranker_scores_to_one_range():
    assert np.allclose(min_max(np.array([4.0, 2.0, 0.0])), [1.0, 0.5, 0.0])
    assert np.allclose(min_max(np.array([3.0, 3.0])), [1.0, 1.0])


def test_assembler_mmr_swaps_a_near_copy_for_a_different_chunk():
    vectors = {A1.chunk_id: [1.0, 0.0], A2.chunk_id: [0.99, 0.141], B1.chunk_id: [0.0, 1.0]}

    def lookup(chunks):
        rows = np.array([vectors[c.chunk_id] for c in chunks])
        return rows / np.linalg.norm(rows, axis=1, keepdims=True)

    candidates = ranked(A1, A2, B1, scores=[3.0, 2.9, 2.0])
    plain = Assembler(top_k=2, pool=3).assemble(candidates, lookup)
    diverse = Assembler(top_k=2, pool=3, mmr=0.5).assemble(candidates, lookup)
    assert plain.chunks == [A1, A2]
    assert diverse.chunks == [A1, B1] and diverse.notes["mmr_swapped"] == 1
    with pytest.raises(ValueError, match="vectors"):
        Assembler(top_k=2, pool=3, mmr=0.5).assemble(candidates)


def test_pool_below_top_k_is_refused():
    with pytest.raises(ValueError, match="pool"):
        Assembler(top_k=5, pool=3)


# --- In the pipeline --------------------------------------------------------------------


def offline_config(tmp_path: Path, **overrides) -> PipelineConfig:
    settings = {
        "name": "test",
        "index_dir": tmp_path / "index",
        "chunker": {"name": "fixed", "params": {"size": 6}},
        "embedder": {"name": "hashing", "params": {"dimensions": 256}},
        "store": {"name": "numpy"},
        "top_k": 3,
        "generator": {"name": "echo"},
    }
    return PipelineConfig.model_validate(settings | overrides)


DOCS = [
    Document("a", TEXT["a"] + " The crossing resumed on Wednesday morning.", META["a"]),
    Document("b", TEXT["b"], META["b"]),
]


def test_a_config_without_context_keeps_the_old_prompt(tmp_path):
    pipeline = Pipeline(offline_config(tmp_path))
    pipeline.ingest(DOCS)
    trace = pipeline.ask("Why was the ferry late?")
    assert trace.context == [c.chunk for c in trace.candidates]
    assert trace.user_prompt == stage1_prompt(trace.question, trace.context)
    assert trace.system_prompt == SYSTEM_PROMPT


def test_pipeline_assembles_with_headers_merging_and_mmr_from_stored_vectors(tmp_path):
    context = {"name": "assembler", "params": {
        "pool": 6, "dedupe": True, "mmr": 0.7, "merge": True, "header": True,
        "order": "chronological"}}
    pipeline = Pipeline(offline_config(tmp_path, context=context))
    pipeline.ingest(DOCS)
    trace = pipeline.ask("Why was the ferry late?")
    # Retrieved to the pool (or all there is), not to top_k.
    assert len(trace.candidates) == min(6, len(pipeline.store)) > 3
    assert sum(len(p.chunks) for p in trace.passages) == 3
    assert all(p.header for p in trace.passages)
    assert trace.system_prompt == HEADER_SYSTEM_PROMPT
    assert trace.assembly["order"] == "chronological"
    dates = [p.metadata["published_at"] for p in trace.passages]
    assert dates == sorted(dates)


def test_stage8_configs_change_only_the_prompt():
    best = load_config(REPO_ROOT / "configs/stage06_rerank/qwen3-100.yaml", PipelineConfig)
    paths = sorted((REPO_ROOT / "configs" / "stage08_context").glob("*.yaml"))
    assert paths
    for path in paths:
        cfg = load_config(path, PipelineConfig)
        assert cfg.name == f"ctx-{path.stem}"
        changed = {k for k, v in cfg.model_dump().items() if v != best.model_dump()[k]}
        allowed = {"name", "top_k", "context"}
        if "luna" in path.stem:  # the Stage 6 short-prompt candidate swaps the reranker
            allowed.add("reranker")
        assert changed <= allowed, (path.name, changed)
        if cfg.context:
            Assembler(top_k=cfg.top_k, **cfg.context.params)  # the params are valid
