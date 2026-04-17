"""
Full document ingestion pipeline: parse -> chunk -> embed -> store

Re-ingesting the same filename is idempotent: old chunks are deleted, new ones replace them atomically.
A simple rag_documents.json registry lives in data/ for fast document listing without querying ChromaDB.
UUIDs are preserved on re-ingest (same file path -> same document ID) so external references remain valid.

Pipeline:-
  Raw bytes / file path ->
  document_parser.parse()    Extract text, detect format ->
   chunker.chunk_text()       Split into overlapping chunks ->
    embedder.embed_many()      Vectorise all chunks in parallel batches ->
     vector_store.add_chunks()  Persist to ChromaDB

Each document gets a UUID. Re-ingesting the same file path replaces all
its chunks atomically (delete old -> insert new).

Ingestion metadata:-
    Document metadata is stored with every chunk so retrieval results can
    be traced back to their source. Additionally, a document registry is
    kept in a simple JSON file for fast listing without querying ChromaDB.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from rag.chunker import chunk_text
from rag.document_parser import parse, ParsedDocument, DocumentParseError
from rag.embedder import get_embedder
from rag.vector_store import get_vector_store
from utils.logger import get_logger
from utils.paths import documents_dir, get_data_dir

log = get_logger(__name__)

_REGISTRY_FILE = get_data_dir() / "rag_documents.json"


@dataclass
class IngestedDocument:
    document_id: str
    title: str
    source_path: str
    file_type: str
    collection: str
    chunk_count: int
    char_count: int
    ingested_at: float


class Ingester:
    """Orchestrates the full document ingestion pipeline."""

    async def ingest_bytes(
        self,
        filename: str,
        data: bytes,
        collection: str = "default",
        chunk_chars: int | None = None,
        overlap_chars: int | None = None,
    ) -> IngestedDocument:
        """
        Ingest a document from raw bytes.

        Args:
            filename:     Original filename (used to detect format and as title).
            data:         Raw file bytes.
            collection:   Target knowledge base.
            chunk_chars:  Override default chunk size from settings.
            overlap_chars: Override default overlap from settings.

        Returns:
            IngestedDocument with metadata about what was stored.
        """
        from config import get_settings
        settings = get_settings()

        c_size = chunk_chars or settings.rag_chunk_chars
        c_overlap = overlap_chars or settings.rag_overlap_chars

        # 1. parse
        path = documents_dir() / filename
        parsed = parse(path=path, raw_bytes=data)
        log.info(
            "rag_parse",
            filename=filename,
            chars=parsed.char_count,
            file_type=parsed.file_type,
        )

        # 2. save original file to disk
        path.write_bytes(data)

        # 3. assign or reuse document ID,
        # if a document with this path already exists, reuse its ID (update)
        registry = _load_registry()
        existing = next(
            (d for d in registry if d["source_path"] == str(path)), None
        )
        document_id = existing["document_id"] if existing else str(uuid.uuid4())

        # 4. delete old chunks if re-ingesting
        if existing:
            await get_vector_store().delete_document(document_id, collection)
            log.info("rag_reingest", document_id=document_id[:8], filename=filename)

        # 5. chunk
        meta_base = {
            "document_id": document_id,
            "title":       parsed.title,
            "source_path": str(path),
            "file_type":   parsed.file_type,
            "collection":  collection,
        }
        chunks = chunk_text(
            text=parsed.text,
            document_id=document_id,
            chunk_chars=c_size,
            overlap_chars=c_overlap,
            metadata=meta_base,
        )
        if not chunks:
            raise DocumentParseError(f"Document '{filename}' produced no chunks after parsing.")

        log.info("rag_chunk", document_id=document_id[:8], chunks=len(chunks))

        # 6. embed all chunks
        t0 = time.perf_counter()
        embedder = get_embedder()
        texts = [c.text for c in chunks]
        vectors = await embedder.embed_many(texts)
        embed_ms = round((time.perf_counter() - t0) * 1000)
        log.info("rag_embed", chunks=len(chunks), embed_ms=embed_ms)

        # 7. build metadata per chunk (including total_chunks now known) -
        chunk_ids = [c.id for c in chunks]
        metadatas = [
            {
                **meta_base,
                "chunk_index":  c.chunk_index,
                "total_chunks": c.total_chunks,
                "char_start":   c.char_start,
            }
            for c in chunks
        ]

        # 8. store in vector DB
        await get_vector_store().add_chunks(
            chunk_ids=chunk_ids,
            texts=texts,
            embeddings=vectors,
            metadatas=metadatas,
            collection=collection,
        )

        # 9. update registry
        doc = IngestedDocument(
            document_id=document_id,
            title=parsed.title,
            source_path=str(path),
            file_type=parsed.file_type,
            collection=collection,
            chunk_count=len(chunks),
            char_count=parsed.char_count,
            ingested_at=time.time(),
        )
        _save_to_registry(doc, registry)

        log.info(
            "rag_ingest_complete",
            document_id=document_id[:8],
            title=parsed.title,
            chunks=len(chunks),
            chars=parsed.char_count,
        )
        return doc

    async def delete_document(
        self, document_id: str, collection: str = "default"
    ) -> bool:
        """Delete a document and all its chunks. Returns True if found."""
        registry = _load_registry()
        entry = next((d for d in registry if d["document_id"] == document_id), None)
        if not entry:
            return False

        await get_vector_store().delete_document(document_id, collection)

        # delete the original file if it exists
        src = Path(entry.get("source_path", ""))
        if src.exists():
            src.unlink()

        # remove from registry
        registry = [d for d in registry if d["document_id"] != document_id]
        _REGISTRY_FILE.write_text(json.dumps(registry, indent=2))
        log.info("rag_delete", document_id=document_id[:8])
        return True

    async def list_documents(self, collection: str = "default") -> list[dict]:
        """List all ingested documents for a collection."""
        registry = _load_registry()
        return [d for d in registry if d.get("collection") == collection]


# registry helpers (simple JSON file)

def _load_registry() -> list[dict]:
    if not _REGISTRY_FILE.exists():
        return []
    try:
        return json.loads(_REGISTRY_FILE.read_text())
    except Exception:
        return []


def _save_to_registry(doc: IngestedDocument, registry: list[dict]) -> None:
    doc_dict = asdict(doc)
    # replace existing entry or append
    updated = [d for d in registry if d["document_id"] != doc.document_id]
    updated.append(doc_dict)
    _REGISTRY_FILE.write_text(json.dumps(updated, indent=2))


# module-level singleton

_ingester: Ingester | None = None


def get_ingester() -> Ingester:
    if _ingester is None:
        raise RuntimeError("Ingester not initialised.")
    return _ingester


def init_ingester() -> Ingester:
    global _ingester
    _ingester = Ingester()
    return _ingester