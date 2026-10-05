"""The Stage 1 app: ingest files, ask questions, and see what was retrieved.

    streamlit run app/streamlit_app.py -- --config configs/baseline.yaml

Each question is answered on its own; the chat has no memory of earlier turns
(follow-up questions arrive in Stage 7).
"""

import argparse
import hashlib
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from ragbasics.cli import DEFAULT_DOCUMENTS, LEDGER, source_label
from ragbasics.config import PipelineConfig, load_config
from ragbasics.eval.dataset import read_documents
from ragbasics.pipeline import Pipeline
from ragbasics.types import Document, Trace

CORPUS = "MultiHop-RAG news corpus"
UPLOADS = "Uploaded files"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    return parser.parse_args()


@st.cache_resource
def get_pipeline(config_path: str, collection: str) -> Pipeline:
    """One pipeline per collection, kept across reruns. Each has its own index on disk."""
    cfg = load_config(config_path, PipelineConfig)
    index_dir = cfg.index_dir if collection == CORPUS else Path(f"{cfg.index_dir}_uploads")
    pipeline = Pipeline(cfg, index_dir=index_dir, ledger=LEDGER)
    pipeline.load()
    return pipeline


def uploaded_document(name: str, data: bytes) -> Document:
    """The id includes a hash of the content, so a changed file is ingested as a new document."""
    text = data.decode("utf-8", errors="replace")
    digest = hashlib.sha256(data).hexdigest()[:8]
    return Document(doc_id=f"upload_{digest}", text=text, metadata={"title": name})


def ingest_with_progress(pipeline: Pipeline, documents: list[Document]) -> None:
    bar = st.progress(0.0, text="Embedding chunks")
    report = pipeline.ingest(documents, progress=lambda done, total: bar.progress(done / total))
    bar.empty()
    st.success(
        f"Indexed {report.documents} documents as {report.chunks} chunks "
        f"({report.skipped} already indexed). Cost ${report.cost_usd:.4f}."
    )


def render_trace(trace: Trace) -> None:
    with st.expander(f"Retrieved chunks ({len(trace.candidates)})"):
        for c in trace.candidates:
            chunk = c.chunk
            st.markdown(f"**{c.rank}. score {c.score:.3f}** · {source_label(chunk.metadata)}")
            st.caption(f"{chunk.chunk_id} · characters {chunk.start_char} to {chunk.end_char}")
            st.text(chunk.text)
    with st.expander("Prompt sent to the generator"):
        st.caption("System")
        st.text(trace.system_prompt)
        st.caption("User")
        st.text(trace.user_prompt)
    timings = " · ".join(f"{step} {s * 1000:.0f} ms" for step, s in trace.timings.items())
    st.caption(
        f"{timings} · {trace.usage.get('input_tokens', 0)} tokens in, "
        f"{trace.usage.get('output_tokens', 0)} out · ${trace.cost_usd:.4f} · "
        f"{trace.models.get('embedder')} + {trace.models.get('generator')}"
    )


def sidebar(config_path: str) -> Pipeline:
    with st.sidebar:
        st.header("Collection")
        collection = st.radio("Ask questions over", [CORPUS, UPLOADS], label_visibility="collapsed")
        pipeline = get_pipeline(config_path, collection)
        documents = len({chunk.doc_id for chunk in pipeline.store.chunks})
        st.caption(f"{documents} documents, {len(pipeline.store)} chunks in `{pipeline.index_dir}`")

        if collection == CORPUS and len(pipeline.store) == 0:
            st.warning("The corpus is not indexed yet.")
            if not Path(DEFAULT_DOCUMENTS).exists():
                st.caption("Run `make data` first.")
            elif st.button("Build the index (about 3 cents with the baseline config)"):
                ingest_with_progress(pipeline, read_documents(Path(DEFAULT_DOCUMENTS)))

        if collection == UPLOADS:
            st.header("Upload and ingest")
            files = st.file_uploader(
                "Plain text or Markdown", type=["txt", "md"], accept_multiple_files=True
            )
            if files and st.button("Ingest"):
                docs = [uploaded_document(f.name, f.getvalue()) for f in files]
                ingest_with_progress(pipeline, docs)
            st.caption("PDF upload arrives in Stage 10.")

        st.header("Pipeline")
        st.caption(
            f"Config `{pipeline.cfg.name}`: {pipeline.cfg.chunker.name} chunks "
            f"{pipeline.cfg.chunker.params}, {pipeline.embedder.model}, "
            f"top {pipeline.cfg.top_k}, {pipeline.generator.model}"
        )
    st.session_state.collection = collection
    return pipeline


def main() -> None:
    load_dotenv()
    args = parse_args()
    st.set_page_config(page_title="RAG Basics", layout="wide")
    st.title("RAG Basics")
    pipeline = sidebar(args.config)

    history: list[tuple[str, Trace]] = st.session_state.setdefault("history", [])
    for collection, trace in history:
        with st.chat_message("user"):
            st.write(trace.question)
        with st.chat_message("assistant"):
            st.write(trace.answer)
            st.caption(f"Collection: {collection}")
            render_trace(trace)

    question = st.chat_input("Ask a question. Each one is answered on its own.")
    if not question:
        return
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Retrieving and answering"):
                trace = pipeline.ask(question)
        except Exception as error:  # shown to the user: missing key, empty index, API error
            st.error(f"{type(error).__name__}: {error}")
            return
        st.write(trace.answer)
        render_trace(trace)
    history.append((st.session_state.collection, trace))


main()
