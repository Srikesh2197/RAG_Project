"""Library equivalents of two hand-written context-assembly steps (Stage 8).

    pip install -e ".[frameworks]"

Maximal marginal relevance: `langchain_core.vectorstores.utils.maximal_marginal_relevance`
against `ragbasics.context.mmr.mmr`.

- The library takes a query vector and the candidates' vectors and computes both terms
  itself: relevance is the cosine between query and candidate, redundancy the cosine
  between candidates. The hand-written function takes the two as inputs, so relevance
  can be anything. Here it is the reranker's score scaled to [0, 1], because the reranker
  ranks far better than the query's cosine (Stage 6), and MMR should trade against the
  best relevance signal there is.
- Given the cosine as relevance, both pick the same items in the same order
  (tests/test_context_parity.py).
- In a LangChain vector store, MMR runs on the first stage's `fetch_k` results, before
  any reranker. Behind a reranker it has to be called by hand, as it is here.

Ordering: `LongContextReorder` (langchain-community) against
`ragbasics.context.assemble.order_edges`.

- The same rule: walk the list from the weakest up and put passages alternately at the
  front and the back. With an odd count the best is first, with an even count last.
- `langchain-community` is being sunset by its maintainers (a deprecation warning on
  import), which is one more reason not to put it on the core path for six lines of code.
"""

from typing import Any

import numpy as np


def langchain_mmr(query: np.ndarray, vectors: np.ndarray, k: int, lam: float) -> list[int]:
    from langchain_core.vectorstores.utils import maximal_marginal_relevance

    return maximal_marginal_relevance(query, list(vectors), lambda_mult=lam, k=k)


def langchain_reorder(items: list[Any]) -> list[Any]:
    """Reorder anything by wrapping it in a Document and reading it back."""
    import warnings

    from langchain_core.documents import Document

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from langchain_community.document_transformers import LongContextReorder

    docs = [Document(page_content=str(i), metadata={"i": i}) for i in range(len(items))]
    reordered = LongContextReorder().transform_documents(docs)
    return [items[d.metadata["i"]] for d in reordered]
