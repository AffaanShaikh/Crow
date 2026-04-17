"""
FastAPI app entry point,
    startup sequence:
        1. Logging configured
        2. LLM client initialised (connects to llama-server)
        3. Context manager initialised
        4. ASR model loaded (if enabled)
        5. TTS model loaded (if enabled)
        6. Tool registry initialised + MCP agent loop wired (if enabled)
        7. Routes registered w/ Middleware (CORS, request logging, error handling)

Run with:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import structlog

from config import get_settings
from llm.client import init_llm_client, shutdown_llm_client
from memory.context_manager import init_context_manager
from api.routes import chat, health
from utils.logger import get_logger, setup_logging

settings = get_settings()
setup_logging(log_level=settings.log_level, json_logs=settings.json_logs)
log = get_logger(__name__)


# lifespan of the app.
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """initialise and shut down all resources"""
    log.info(
        "app_starting",
        name=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
    )

    # initialise the singletons
    llm_client = init_llm_client()
    init_context_manager()

    # to verify llama.cpp server is reachable at startup
    healthy, latency = await llm_client.health_check()
    if healthy:
        log.info("llm_backend_ready", latency_ms=round(latency, 1))
    else:
        log.warning(
            "llm_backend_unreachable",
            url=settings.llm_base_url,
            hint="Make sure llama-server is running: ollama serve",
        )

    # Wake word detector
    if getattr(settings, "wake_word_enabled", False):
        from audio.wake_word import init_wake_detector
        detector = init_wake_detector()
        loop = asyncio.get_event_loop()
        detector.start(loop)
        log.info("wake_word_ready", model=settings.wake_word_model)
    else:
        log.info("wake_word_disabled")

    # ASR
    if settings.asr_enabled:
        from audio.asr import init_asr_service
        asr = init_asr_service()
        await asr.load_model()
        log.info("asr_ready", model=settings.asr_model_size)
    else:
        log.info("asr_disabled")

    # TTS
    if settings.tts_enabled:
        from audio.tts import init_tts_service
        tts = init_tts_service()
        await tts.load_model()
        log.info("tts_ready", voice=settings.tts_voice)
    else:
        log.info("tts_disabled")

    # MCP / Agent
    if settings.mcp_enabled:
        from mcp.registry import init_registry
        from mcp.dispatcher import ToolDispatcher
        from mcp.router import init_router
        from mcp.agent_loop import init_agent_loop
        registry = init_registry()
        dispatcher = ToolDispatcher(registry, dry_run=settings.mcp_dry_run)
        router = init_router(llm_client, registry)
        init_agent_loop(llm_client, dispatcher, registry, router)
        log.info("mcp_ready", tools=len(registry), dry_run=settings.mcp_dry_run)
    else:
        log.info("mcp_disabled")

    # RAG
    if settings.rag_enabled:
        from rag.vector_store import init_vector_store
        from rag.embedder import init_embedder
        from rag.ingester import init_ingester
        from rag.retriever import init_retriever
        await init_vector_store()
        init_embedder()
        init_ingester()
        init_retriever()
        # verify embedding model is available
        from rag.embedder import get_embedder
        ready, msg = await get_embedder().health_check()
        if ready:
            log.info("rag_ready", embedding_model=settings.rag_embedding_model)
        else:
            log.warning(
                "rag_embedding_model_unavailable",
                message=msg,
                hint=f"Run: ollama pull {settings.rag_embedding_model}",
            )
    else:
        log.info("rag_disabled")

    log.info("app_ready", host=settings.host, port=settings.port)
    yield  # <- application runs here

    log.info("app_shutting_down")
    if getattr(settings, "wake_word_enabled", False):
        try:
            from audio.wake_word import get_wake_detector
            get_wake_detector().stop()
        except Exception:
            pass
    shutdown_llm_client()
    log.info("app_stopped")


# the app. factory
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="My FastAPI app.",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)


# the middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def request_logging_middleware(request: Request, call_next) -> Response:
    """logs every request with timing and binds a request_id for log correlation"""
    request_id = str(uuid.uuid4())[:8]
    structlog.contextvars.bind_contextvars(request_id=request_id)

    t0 = time.perf_counter()
    response: Response = await call_next(request)
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

    # no spamming logs with health-check noise 
    if request.url.path not in ("/api/v1/health", "/"):
        log.info("http_request", method=request.method, path=request.url.path,
                 status=response.status_code, elapsed_ms=elapsed_ms)
    response.headers["X-Request-ID"] = request_id
    structlog.contextvars.clear_contextvars()
    return response



@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    log.exception("unhandled_exception", path=request.url.path, error=str(exc))
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred.", "type": type(exc).__name__},
    )


# routers - registration order matters for OpenAPI docs but not for routing

# /chat   - session management + non-streaming fallback (always registered)
# /agent  - unified streaming endpoint, always registered when MCP is enabled
#           (replaces the old /chat/stream - the agent route handles both
#            conversational and tool-use paths via the ToolRouter)
# /audio  - ASR WebSocket + TTS HTTP (registered when either flag is enabled)
# /health - liveness probe

app.include_router(chat.router, prefix="/api/v1")
app.include_router(health.router, prefix="/api/v1")

# Auth routes: (Google, Spotify, ..), 
# OAuth (always registered - no feature flag, since auth must work even if the tools themselves are disabled)
from api.routes.auth import router as auth_router
app.include_router(auth_router, prefix="/api/v1")

# Agent route: always registered when MCP is enabled,
# ToolRouter inside handles Path A (direct persona) vs Path B (tools),
# /chat/stream no longer registered - /agent/stream the only stream endpoint
if settings.mcp_enabled:
    from api.routes.agent import router as agent_router
    app.include_router(agent_router, prefix="/api/v1")
    
# audio routes (registered only if enabled to keep import side-effects lazy)
if settings.asr_enabled or settings.tts_enabled or getattr(settings, "wake_word_enabled", False):
    from api.routes.audio import router as audio_router
    from api.routes.wake_routes import register_wake_routes
    register_wake_routes(audio_router)
    app.include_router(audio_router, prefix="/api/v1")

# RAG routes
if settings.rag_enabled:
    from api.routes.rag import router as rag_router
    app.include_router(rag_router, prefix="/api/v1")


@app.get("/", include_in_schema=False)
async def root():
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/api/v1/health",
        "features": {
            "asr": settings.asr_enabled,
            "tts": settings.tts_enabled,
            "mcp": settings.mcp_enabled,
        },
    }




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.environment == "development",
        log_config=None, # let structlog handle all logging
    )
