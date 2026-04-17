"""
ChromaDB vector store wrapper, uses ChromaDB in embedded (serverless) mode, no separate process needed.
Data persisted to disk at data/chromadb/.

Each "knowledge base" is a separate ChromaDB collection. Default collection: "default"
Users can have multiple knowledge bases (ex. "work", "personal", "codebase")
each with their own documents and retrieval scope.

METADATA STORED PER CHUNK:-
  document_id   : parent document UUID
  chunk_index   : position within document
  total_chunks  : total chunks in document
  source_path   : original file path
  title         : document title
  file_type     : pdf / txt / docx / etc.
  collection    : collection name (redundant but useful for debugging)

ChromaDB distance func., we are using cosine distance (default). With normalised embeddings (which Ollama
produces), cosine similarity = 1 - cosine_distance. Lower distance = more similar.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from typing import Any

from utils.logger import get_logger
from utils.paths import chromadb_dir

log = get_logger(__name__)

DEFAULT_COLLECTION = "default"


@dataclass
class SearchResult:
    chunk_id: str
    text: str
    score: float        # similarity score 0-1 (higher = more similar)
    document_id: str
    title: str
    source_path: str
    chunk_index: int
    total_chunks: int
    metadata: dict[str, Any]


class VectorStore:
    """
    Async wrapper around a ChromaDB PersistentClient.

    All blocking ChromaDB operations are run in a thread pool so they
    don't block the FastAPI event loop.
    """

    def __init__(self) -> None:
        self._client = None
        self._collections: dict[str, Any] = {}

    def _init_client(self) -> None:
        """Initialise ChromaDB (blocking - called from thread pool)."""
        import chromadb
        path = str(chromadb_dir())
        self._client = chromadb.PersistentClient(path=path)
        log.info("chromadb_init", path=path)

    async def ready(self) -> None:
        """Ensure the client is initialised. Call at startup."""
        if self._client is None:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._init_client)

    # collection management

    def _get_collection(self, name: str = DEFAULT_COLLECTION):
        """Get or create a collection (blocking)."""
        if name not in self._collections:
            self._collections[name] = self._client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collections[name]

    async def get_collection(self, name: str = DEFAULT_COLLECTION):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_collection, name)

    async def list_collections(self) -> list[str]:
        loop = asyncio.get_event_loop()
        cols = await loop.run_in_executor(None, self._client.list_collections)
        return [c.name for c in cols]

    # insert

    async def add_chunks(
        self,
        chunk_ids: list[str],
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
        collection: str = DEFAULT_COLLECTION,
    ) -> None:
        """
        Insert chunks into the vector store.
        Upserts (overwrites) if chunk IDs already exist.
        """
        if not chunk_ids:
            return

        col = await self.get_collection(collection)
        loop = asyncio.get_event_loop()

        def _upsert():
            col.upsert(
                ids=chunk_ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
            )

        await loop.run_in_executor(None, _upsert)
        log.info("vector_store_add", count=len(chunk_ids), collection=collection)

    # search

    async def search(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        collection: str = DEFAULT_COLLECTION,
        where: dict | None = None,
    ) -> list[SearchResult]:
        """
        Search for the n_results most similar chunks to query_embedding.

        Args:
            query_embedding: The query vector (same dimensions as stored vectors).
            n_results:       Number of results to return.
            collection:      Which knowledge base to search.
            where:           Optional ChromaDB metadata filter
                             e.g. {"document_id": {"$eq": "some-id"}}

        Returns:
            List of SearchResult, sorted by similarity descending.
        """
        col = await self.get_collection(collection)
        loop = asyncio.get_event_loop()

        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        def _query():
            return col.query(**kwargs)

        results = await loop.run_in_executor(None, _query)

        search_results: list[SearchResult] = []
        if not results or not results["ids"] or not results["ids"][0]:
            return search_results

        for i, chunk_id in enumerate(results["ids"][0]):
            distance = results["distances"][0][i]
            # ChromaDB cosine distance: 0 = identical, 2 = opposite,
            # convert to similarity score 0-1
            score = max(0.0, 1.0 - distance / 2.0)

            meta = results["metadatas"][0][i] or {}
            text = results["documents"][0][i] or ""

            search_results.append(SearchResult(
                chunk_id=chunk_id,
                text=text,
                score=score,
                document_id=meta.get("document_id", ""),
                title=meta.get("title", "Unknown"),
                source_path=meta.get("source_path", ""),
                chunk_index=meta.get("chunk_index", 0),
                total_chunks=meta.get("total_chunks", 1),
                metadata=meta,
            ))

        return search_results

    # document management

    async def delete_document(
        self,
        document_id: str,
        collection: str = DEFAULT_COLLECTION,
    ) -> int:
        """Delete all chunks belonging to a document. Returns number deleted."""
        col = await self.get_collection(collection)
        loop = asyncio.get_event_loop()

        def _delete():
            col.delete(where={"document_id": {"$eq": document_id}})

        await loop.run_in_executor(None, _delete)
        log.info("vector_store_delete_document", document_id=document_id[:8])
        return 0 # ChromaDB delete doesn't return count

    async def get_document_list(
        self,
        collection: str = DEFAULT_COLLECTION,
    ) -> list[dict[str, Any]]:
        """
        Return a deduplicated list of documents stored in a collection.
        Groups by document_id, returns one entry per document.
        """
        col = await self.get_collection(collection)
        loop = asyncio.get_event_loop()

        def _get_all():
            return col.get(include=["metadatas"])

        raw = await loop.run_in_executor(None, _get_all)
        if not raw or not raw["metadatas"]:
            return []

        seen: dict[str, dict] = {}
        for meta in raw["metadatas"]:
            if not meta:
                continue
            doc_id = meta.get("document_id", "")
            if doc_id and doc_id not in seen:
                seen[doc_id] = {
                    "document_id": doc_id,
                    "title":       meta.get("title", "Unknown"),
                    "source_path": meta.get("source_path", ""),
                    "file_type":   meta.get("file_type", ""),
                    "total_chunks": meta.get("total_chunks", 0),
                    "collection":  collection,
                }
        return list(seen.values())

    async def collection_stats(self, collection: str = DEFAULT_COLLECTION) -> dict:
        """Return basic stats about a collection."""
        col = await self.get_collection(collection)
        loop = asyncio.get_event_loop()
        count = await loop.run_in_executor(None, col.count)
        docs = await self.get_document_list(collection)
        return {
            "collection":    collection,
            "total_chunks":  count,
            "total_documents": len(docs),
        }


# singleton

_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    if _store is None:
        raise RuntimeError("VectorStore not initialised.")
    return _store


async def init_vector_store() -> VectorStore:
    global _store
    _store = VectorStore()
    await _store.ready()
    log.info("vector_store_ready", path=str(chromadb_dir()))
    return _store