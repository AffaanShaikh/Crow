"""
Retrieval pipeline: query -> embed -> search -> rerank -> context.

2stage retrieval:-
────────────────────
Stage 1 - Dense retrieval (embedding similarity):
  Query text -> embed -> cosine search in ChromaDB -> top-K candidates
  Fast, high recall, but sometimes noisy.

Stage 2 - Cross-encoder reranking (optional):
  Each (query, candidate) pair -> cross-encoder relevance score -> reorder
  Slower but much more precise: the cross-encoder reads both texts together
  rather than comparing independent vectors.

Reranking is off by default (RAG_RERANKER_ENABLED=false) because it requires sentence-transformers 
 and adds ~100ms. When enabled, we fetch kx3 candidates from ChromaDB and rerank to top k -
 the extra candidates make reranking worthwhile.
When reranking is enabled, we retrieve 3x more candidates from Stage 1
(rag_retrieval_k * 3) and then rerank to return the top rag_retrieval_k.

Reranker model:-
Default: cross-encoder/ms-marco-MiniLM-L-6-v2 (via sentence-transformers)
~70 MB, runs on CPU, ~20ms per (query, passage) pair.
Set RAG_RERANKER_ENABLED=false in .env to skip reranking.

Foramtting context:-
Retrieved chunks are assembled into a context string injected into
build_messages(rag_context=...). The format:

  [Source: document_title.pdf]
  <chunk text>

  [Source: another_document.txt]
  <chunk text>
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from rag.embedder import get_embedder
from rag.vector_store import SearchResult, get_vector_store
from utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class RetrievalResult:
    results: list[SearchResult]
    query: str
    collection: str
    retrieval_ms: float
    reranked: bool
    context_str: str        # formatted for injection into prompt


class Retriever:
    """
    Query -> embed -> search -> (optional rerank) -> context string.

    Usage::

        retriever = Retriever()
        result = await retriever.retrieve("What is the refund policy?")
        # result.context_str is ready for build_messages(rag_context=...)
    """

    def __init__(self) -> None:
        self._reranker = None
        self._reranker_attempted = False

    async def retrieve(
        self,
        query: str,
        collection: str = "default",
        k: int | None = None,
        score_threshold: float | None = None,
    ) -> RetrievalResult:
        """
        Run the full retrieval pipeline for a query.

        Args:
            query:           The user's question or search text.
            collection:      Which knowledge base to search.
            k:               Number of results. Defaults to settings.rag_retrieval_k.
            score_threshold: Minimum similarity score (0-1). Chunks below this
                             are excluded. Defaults to settings.rag_score_threshold.

        Returns:
            RetrievalResult with retrieved chunks and a formatted context string.
        """
        from config import get_settings
        settings = get_settings()

        k = k or settings.rag_retrieval_k
        threshold = score_threshold if score_threshold is not None else settings.rag_score_threshold
        reranking_enabled = settings.rag_reranker_enabled

        t0 = time.perf_counter()

        # Stage 1: dense retrieval
        # fetch more candidates when reranking so the reranker has more to work with
        fetch_k = k * 3 if reranking_enabled else k

        embedder = get_embedder()
        query_vector = await embedder.embed(query)

        candidates = await get_vector_store().search(
            query_embedding=query_vector,
            n_results=min(fetch_k, 50),     # ChromaDB cap
            collection=collection,
        )

        # filter by score threshold
        candidates = [r for r in candidates if r.score >= threshold]

        if not candidates:
            elapsed = round((time.perf_counter() - t0) * 1000)
            log.info("rag_retrieve_empty", query_preview=query[:60], elapsed_ms=elapsed)
            return RetrievalResult(
                results=[],
                query=query,
                collection=collection,
                retrieval_ms=elapsed,
                reranked=False,
                context_str="",
            )

        # Stage 2: reranking (optional)
        reranked = False
        if reranking_enabled and len(candidates) > 1:
            try:
                candidates = await self._rerank(query, candidates, top_k=k)
                reranked = True
            except Exception as exc:
                log.warning("rag_rerank_failed", error=str(exc), fallback="using dense results")
                # fall back to top-k from Stage 1
                candidates = candidates[:k]
        else:
            candidates = candidates[:k]

        elapsed = round((time.perf_counter() - t0) * 1000)
        log.info(
            "rag_retrieve",
            query_preview=query[:60],
            results=len(candidates),
            reranked=reranked,
            elapsed_ms=elapsed,
            top_score=round(candidates[0].score, 3) if candidates else 0,
        )

        context_str = _format_context(candidates)
        return RetrievalResult(
            results=candidates,
            query=query,
            collection=collection,
            retrieval_ms=elapsed,
            reranked=reranked,
            context_str=context_str,
        )

    async def _rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        """
        Cross-encoder reranking. Loads the reranker model on first use.
        Runs in a thread pool (CPU-bound).
        """
        from config import get_settings
        model_name = get_settings().rag_reranker_model

        if not self._reranker_attempted:
            self._reranker_attempted = True
            try:
                from sentence_transformers import CrossEncoder
                self._reranker = CrossEncoder(model_name)
                log.info("reranker_loaded", model=model_name)
            except ImportError:
                log.warning(
                    "reranker_unavailable",
                    hint="pip install sentence-transformers",
                )
                raise RuntimeError("sentence-transformers not installed")

        if self._reranker is None:
            raise RuntimeError("Reranker failed to load on previous attempt")

        pairs = [(query, c.text) for c in candidates]
        loop = asyncio.get_event_loop()

        scores = await loop.run_in_executor(None, self._reranker.predict, pairs)

        scored = sorted(
            zip(scores, candidates),
            key=lambda x: x[0],
            reverse=True,
        )
        return [c for _, c in scored[:top_k]]


# context formatting

def _format_context(results: list[SearchResult]) -> str:
    """
    Format retrieved chunks into a context string for prompt injection.

    Groups chunks by source document and includes a citation header.
    """
    if not results:
        return ""

    parts: list[str] = []
    for r in results:
        source = r.title or r.source_path.split("/")[-1] if r.source_path else "Unknown"
        # if the document has multiple chunks in results - show chunk position 
        same_doc = [x for x in results if x.document_id == r.document_id]
        if len(same_doc) > 1:
            pos = f"part {r.chunk_index + 1}/{r.total_chunks}"
            parts.append(f"[Source: {source} - {pos}]\n{r.text}")
        else:
            parts.append(f"[Source: {source}]\n{r.text}")

    return "\n\n".join(parts)


# module-level singleton

_retriever: Retriever | None = None


def get_retriever() -> Retriever:
    if _retriever is None:
        raise RuntimeError("Retriever not initialised.")
    return _retriever


def init_retriever() -> Retriever:
    global _retriever
    _retriever = Retriever()
    return _retriever