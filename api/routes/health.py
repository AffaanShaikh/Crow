"""
health check routes, useful for frontend status indication and future monitoring and debugging
GET /health - used by load balancers, monitoring, and the frontend status indicator.
"""

from fastapi import APIRouter, Depends
from config import get_settings, Settings
from llm.client import LLMClient, get_llm_client
from models.schemas import ComponentHealth, HealthResponse, HealthStatus
from utils.logger import get_logger

log = get_logger(__name__)
router = APIRouter(prefix="/health", tags=["health"])


async def _check_vector_db(settings) -> ComponentHealth:
    """Check RAG vector DB and embedding model health."""
    if not settings.rag_enabled:
        return ComponentHealth(status=HealthStatus.OK, detail="disabled")
    try:
        from rag.vector_store import get_vector_store
        from rag.embedder import get_embedder
        store = get_vector_store()
        cols = await store.list_collections()
        embedder = get_embedder()
        emb_ok, emb_msg = await embedder.health_check()
        detail = f"ChromaDB: {len(cols)} collection(s) | Embedder: {emb_msg}"
        st = HealthStatus.OK if emb_ok else HealthStatus.DEGRADED
        return ComponentHealth(status=st, detail=detail)
    except RuntimeError:
        return ComponentHealth(status=HealthStatus.DEGRADED, detail="not initialised")
    except Exception as exc:
        return ComponentHealth(status=HealthStatus.DOWN, detail=str(exc)[:80])


@router.get("", response_model=HealthResponse)
async def health_check(
    llm: LLMClient = Depends(get_llm_client),
    settings: Settings = Depends(get_settings),
) -> HealthResponse:
    """
    checks the llama.cpp server liveness and reports feature flag states
    Returns:
        overall system health and per-component status
    """
    # llm backend health check
    llm_healthy, llm_latency = await llm.health_check()

    components = {
        "llm_backend": ComponentHealth(
            status=HealthStatus.OK if llm_healthy else HealthStatus.DOWN,
            detail=settings.llm_base_url,
            latency_ms=round(llm_latency, 1),
        ),
        "vector_db": await _check_vector_db(settings),
        # "image_gen": ComponentHealth(
        #     status=HealthStatus.OK if settings.avatar_enabled else HealthStatus.DEGRADED,
        #     detail="Gen. AI" if not settings.avatar_enabled else "disabled via flag",
        # ),
    }

    overall = (
        HealthStatus.OK
        if all(c.status == HealthStatus.OK for c in components.values())
        else HealthStatus.DEGRADED
        if llm_healthy
        else HealthStatus.DOWN
    )

    return HealthResponse(
        status=overall,
        version=settings.app_version,
        components=components,
        feature_flags={
            "avatar": settings.avatar_enabled,
            "tts": settings.tts_enabled,
            "asr": settings.asr_enabled,
            "mcp": settings.mcp_enabled,
            "rag": settings.rag_enabled,
            "wake_word":  getattr(settings, "wake_word_enabled", False),
        },
    )
