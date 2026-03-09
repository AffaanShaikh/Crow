"""
FastAPI app for the local AI,
    startup sequence:
        1. Logging configured
        2. LLM client initialised (connects to llama-server)
        3. Context manager initialised
        4. Routes registered
        5. Middleware applied (CORS, request logging, error handling)

Run with:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

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

    log.info("app_ready", host=settings.host, port=settings.port)
    yield  # <- application runs here

    log.info("app_shutting_down")
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

    # don't spam logs with health-check noise
    if request.url.path != "/health":
        log.info(
            "http_request",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            elapsed_ms=elapsed_ms,
        )

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


# routers
app.include_router(chat.router, prefix="/api/v1")
app.include_router(health.router, prefix="/api/v1")


@app.get("/", include_in_schema=False)
async def root():
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/api/v1/health",
    }




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.environment == "development",
        log_config=None,   # let structlog handle all logging
    )
