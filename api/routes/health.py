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
        # stubs for future components
        "vector_db": ComponentHealth(
            status=HealthStatus.OK,
            detail="not initialised",
        ),
        "image_gen": ComponentHealth(
            status=HealthStatus.OK if settings.avatar_enabled else HealthStatus.DEGRADED,
            detail="ComfyUI" if not settings.avatar_enabled else "disabled via flag",
        ),
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
            "rag": settings.rag_enabled,
        },
    )
