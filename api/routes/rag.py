"""
RAG API routes:-
  POST   /rag/documents              - upload and ingest a document
  GET    /rag/documents              - list ingested documents
  DELETE /rag/documents/{doc_id}     - delete a document and its chunks
  POST   /rag/search                 - search the knowledge base directly
  GET    /rag/collections            - list available collections
  GET    /rag/stats                  - collection stats
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import Response
from pydantic import BaseModel

from config import get_settings, Settings
from rag.document_parser import SUPPORTED_EXTENSIONS, DocumentParseError
from rag.ingester import get_ingester
from rag.retriever import get_retriever
from utils.logger import get_logger

log = get_logger(__name__)
router = APIRouter(prefix="/rag", tags=["rag"])

_MAX_FILE_BYTES = 50 * 1024 * 1024 # 50 MB per file


# Upload + Ingest

@router.post("/documents", status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    collection: str = Form(default="default"),
    settings: Settings = Depends(get_settings),
) -> dict:
    """
    Upload a document and ingest it into the RAG knowledge base.

    Supported formats: .txt, .md, .pdf, .docx, .html, and most code files.
    Re-uploading a file with the same name replaces the existing entry.

    The ingestion pipeline: parse -> chunk -> embed -> store.
    Returns immediately when embedding is complete.
    """
    if not settings.rag_enabled:
        raise HTTPException(status_code=503, detail="RAG is disabled (RAG_ENABLED=false in .env)")

    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    suffix = "." + file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported file type '{suffix}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            ),
        )

    data = await file.read()
    if len(data) > _MAX_FILE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(data) // 1024} KB). Maximum is 50 MB."
        )
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        ingester = get_ingester()
        doc = await ingester.ingest_bytes(
            filename=file.filename,
            data=data,
            collection=collection,
        )
    except DocumentParseError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except RuntimeError as exc:
        # ex. embedding model not pulled
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        log.exception("rag_upload_error", filename=file.filename, error=str(exc))
        raise HTTPException(status_code=500, detail="Ingestion failed. Check backend logs.")

    return {
        "document_id": doc.document_id,
        "title":        doc.title,
        "file_type":    doc.file_type,
        "chunk_count":  doc.chunk_count,
        "char_count":   doc.char_count,
        "collection":   doc.collection,
        "ingested_at":  doc.ingested_at,
    }


# List documents

@router.get("/documents")
async def list_documents(
    collection: str = "default",
    settings: Settings = Depends(get_settings),
) -> dict:
    """List all ingested documents in a collection."""
    if not settings.rag_enabled:
        raise HTTPException(status_code=503, detail="RAG is disabled")
    docs = await get_ingester().list_documents(collection)
    return {"documents": docs, "total": len(docs), "collection": collection}


# delete document

@router.delete(
    "/documents/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
)
async def delete_document(
    document_id: str,
    collection: str = "default",
    settings: Settings = Depends(get_settings),
) -> Response:
    """Delete a document and all its chunks from the knowledge base."""
    if not settings.rag_enabled:
        raise HTTPException(status_code=503, detail="RAG is disabled")

    found = await get_ingester().delete_document(document_id, collection)
    if not found:
        raise HTTPException(status_code=404, detail=f"Document '{document_id}' not found")

    return Response(status_code=status.HTTP_204_NO_CONTENT)


# direct search

class SearchRequest(BaseModel):
    query: str
    collection: str = "default"
    k: int = 5
    score_threshold: float = 0.3


@router.post("/search")
async def search_knowledge_base(
    req: SearchRequest,
    settings: Settings = Depends(get_settings),
) -> dict:
    """
    Search the knowledge base directly. Returns matching chunks with scores.
    Useful for testing retrieval quality before querying the AI.
    """
    if not settings.rag_enabled:
        raise HTTPException(status_code=503, detail="RAG is disabled")
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        result = await get_retriever().retrieve(
            query=req.query,
            collection=req.collection,
            k=req.k,
            score_threshold=req.score_threshold,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    return {
        "query":          req.query,
        "collection":     req.collection,
        "results":        [
            {
                "chunk_id":    r.chunk_id,
                "text":        r.text[:500] + ("..." if len(r.text) > 500 else ""),
                "score":       round(r.score, 4),
                "title":       r.title,
                "source_path": r.source_path,
                "chunk_index": r.chunk_index,
            }
            for r in result.results
        ],
        "retrieval_ms":   result.retrieval_ms,
        "reranked":       result.reranked,
    }


# stats

@router.get("/stats")
async def collection_stats(
    collection: str = "default",
    settings: Settings = Depends(get_settings),
) -> dict:
    """Return stats for a collection."""
    if not settings.rag_enabled:
        raise HTTPException(status_code=503, detail="RAG is disabled")
    from rag.vector_store import get_vector_store
    return await get_vector_store().collection_stats(collection)


@router.get("/collections")
async def list_collections(settings: Settings = Depends(get_settings)) -> dict:
    """List all available knowledge base collections."""
    if not settings.rag_enabled:
        raise HTTPException(status_code=503, detail="RAG is disabled")
    from rag.vector_store import get_vector_store
    cols = await get_vector_store().list_collections()
    return {"collections": cols}