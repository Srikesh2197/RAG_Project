"""Library equivalents of two hand-written query transformations: LangChain's
`MultiQueryRetriever` and `HypotheticalDocumentEmbedder` (HyDE).

    pip install -e ".[frameworks]"

Both wrap the same three steps the hand-written versions spell out: prompt an LLM with
the question, read queries out of its reply, and search with them.

MultiQueryRetriever against `ragbasics.query.llm.MultiQuery`:

- Parsing. The library splits the reply on newlines and drops empty lines. It keeps
  numbering and bullets as part of the query ("1. Why was..."), keeps repeats, and has
  no limit on how many lines it accepts. The hand-written parser strips, de-duplicates
  and cuts.
- Combining. The library returns the union of every query's results, each document
  once, in the order they were first seen: all of the first query's, then whatever the
  second adds, and so on. There is no ranking across queries, so the list has no
  meaningful order and grows with the number of queries. The hand-written pipeline
  combines the lists by reciprocal rank or by round robin (each list's next best in
  turn, the form Stage 7 chose) and returns a ranked list of the length asked for.
- Failure. A provider's refusal or an empty reply is not handled; the hand-written
  version falls back to the question and records it.

HypotheticalDocumentEmbedder against `ragbasics.query.llm.HyDE`:

- It is an embedder, not a retriever: `embed_query` writes the passage and returns the
  passage's vector, embedded as a document. That is the same idea as passing
  `kind="document"` here. Asked for several passages, it averages their vectors.
- It replaces the dense query only. In a hybrid pipeline the keyword side still needs
  a query, and which one it gets is the caller's decision.

LlamaIndex's `SubQuestionQueryEngine` is the library form of decomposition. It is not
installed here. It differs in kind as well as in detail: it sends each sub-question to a
query engine that retrieves and answers it, then asks the LLM to combine the
sub-answers. The hand-written `Decomposer` stops at retrieval and gives the generator the
original question with the combined passages, which is one generator call and not one
per sub-question.

The LLM's output is not deterministic, so a parity test cannot compare two live runs.
tests/test_query_parity.py gives both sides the same scripted reply and compares what is
deterministic: the queries read out of it, and the chunks retrieved with them.
"""

from typing import Any

from langchain_classic.chains.hyde.base import HypotheticalDocumentEmbedder
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake import FakeListLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever

from ragbasics.pipeline import Pipeline


class PipelineRetriever(BaseRetriever):
    """The repo's retriever behind LangChain's interface: one query in, the top `k`
    chunks out as Documents that carry their chunk id."""

    pipeline: Any
    k: int

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        found = self.pipeline.retriever.retrieve(query, self.k, self.pipeline).candidates
        return [Document(page_content=c.chunk.text, id=c.chunk.chunk_id) for c in found]


class PipelineEmbeddings(Embeddings):
    """The repo's embedder behind LangChain's interface."""

    def __init__(self, embedder: Any):
        self.embedder = embedder

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embedder.embed(texts, kind="document").vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embedder.embed([text], kind="query").vectors[0].tolist()


def multi_query(pipeline: Pipeline, reply: str, k: int, system_prompt: str) -> MultiQueryRetriever:
    """LangChain's multi-query retriever over the repo's retriever, with an LLM that
    always gives `reply`. The question is searched too, as the hand-written version does
    by default."""
    prompt = PromptTemplate.from_template(system_prompt.replace("{", "{{").replace("}", "}}")
                                          + "\n\nQuestion: {question}")
    return MultiQueryRetriever.from_llm(
        retriever=PipelineRetriever(pipeline=pipeline, k=k),
        llm=FakeListLLM(responses=[reply]),
        prompt=prompt,
        include_original=True,
    )


def hyde_vector(pipeline: Pipeline, reply: str, question: str, system_prompt: str) -> list[float]:
    """The vector LangChain's HyDE embedder searches with, for an LLM that always
    gives `reply`."""
    prompt = PromptTemplate.from_template(system_prompt.replace("{", "{{").replace("}", "}}")
                                          + "\n\nQuestion: {QUESTION}")
    embedder = HypotheticalDocumentEmbedder.from_llm(
        FakeListLLM(responses=[reply]),
        PipelineEmbeddings(pipeline.embedder),
        custom_prompt=prompt,
    )
    return embedder.embed_query(question)
