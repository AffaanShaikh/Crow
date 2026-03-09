"""
chat API routes, aka our traffic controller for all things LLM interaction

POST    /chat/stream        - Server-Sent Events streaming (primary endpoint)
POST    /chat               - Non-streaming fallback
GET     /chat/session/{id}  - Session info
DELETE  /chat/session/{id}  - Clear a session
"""
 
import time
import uuid
from collections.abc import AsyncIterator

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from config import get_settings, Settings
from llm.client import LLMClient, get_llm_client
from llm.prompt_builder import build_messages
from memory.context_manager import ContextManager, get_context_manager
from models.schemas import (
    ChatRequest,
    ChatResponse,
    SessionInfo,
    StreamEvent,
    StreamEventType,
)
from utils.logger import get_logger

log = get_logger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


# streaming endpoint using Server-Sent Events (SSE)
@router.post("/stream")
async def chat_stream(
    req: ChatRequest,
    request: Request,
    llm: LLMClient = Depends(get_llm_client),
    ctx: ContextManager = Depends(get_context_manager),
    settings: Settings = Depends(get_settings),
) -> StreamingResponse:
    """
    streams LLM response tokens via Server-Sent Events,
    frontend connects via EventSource and reads 'data:' lines,
    each line is a JSON-encoded StreamEvent

    Events:
      {"type": "delta", "content": "<token>"}
      {"type": "done",  "metadata": {"elapsed_ms": 123}}
      {"type": "error", "error": "<message>"}
    """
    request_id = str(uuid.uuid4())[:8]
    structlog.contextvars.bind_contextvars(
        request_id=request_id,
        session_id=req.session_id,
    )
    log.info("chat_stream_start", message_preview=req.message[:80])

    async def event_generator() -> AsyncIterator[str]:
        """
        - builds the prompt
        - opens a stream to the LLM
        - for each token: checks if client disconnected (avoids wasted compute), yields a formatted SSE line
        - after streaming completes: saves the full turn to memory
        - sends a done event so the frontend knows to stop the cursor
        """
        full_response: list[str] = []
        t0 = time.perf_counter()

        try:
            history, summary = await ctx.get_context(req.session_id)
            messages = build_messages(
                user_message=req.message,
                history_turns=history,
                summary=summary,
            )

            async for token in llm.stream(
                messages,
                temperature=req.temperature,
                max_tokens=req.max_tokens,
            ):
                # to check if client disconnected mid-stream
                if await request.is_disconnected():
                    log.info("client_disconnected_mid_stream")
                    break

                full_response.append(token)
                event = StreamEvent(type=StreamEventType.DELTA, content=token)
                yield _sse(event)

            # persist the completed turn
            complete_response = "".join(full_response)
            if complete_response:
                await ctx.add_turn(
                    session_id=req.session_id,
                    user_message=req.message,
                    assistant_response=complete_response,
                )

            elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
            done_event = StreamEvent(
                type=StreamEventType.DONE,
                session_id=req.session_id,
                metadata={"elapsed_ms": elapsed_ms, "request_id": request_id},
            )
            yield _sse(done_event)
            log.info("chat_stream_done", elapsed_ms=elapsed_ms)

        except Exception as exc:
            log.exception("chat_stream_error", error=str(exc))
            error_event = StreamEvent(
                type=StreamEventType.ERROR,
                error="The model encountered an error. Please try again.",
            )
            yield _sse(error_event)
        finally:
            structlog.contextvars.clear_contextvars()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no", # disable nginx buffering
            "Connection": "keep-alive",
        },
    )


# non-streaming output, for clients that can't handle SSE (curl, Postman)
@router.post("", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    llm: LLMClient = Depends(get_llm_client),
    ctx: ContextManager = Depends(get_context_manager),
    settings: Settings = Depends(get_settings),
) -> ChatResponse:
    """
    non-streaming chat completion,
    returns the full response once generation is complete
    """
    log.info("chat_start", session_id=req.session_id)

    history, summary = await ctx.get_context(req.session_id)
    messages = build_messages(
        user_message=req.message,
        history_turns=history,
        summary=summary,
    )

    text, usage = await llm.complete(
        messages,
        temperature=req.temperature,
        max_tokens=req.max_tokens,
    )

    await ctx.add_turn(
        session_id=req.session_id,
        user_message=req.message,
        assistant_response=text,
    )

    return ChatResponse(
        session_id=req.session_id,
        message=text,
        model=settings.llm_model_name,
        usage=usage,
    )


# session management endpoints
@router.get("/session/{session_id}", response_model=SessionInfo)
async def get_session(
    session_id: str,
    ctx: ContextManager = Depends(get_context_manager),
    settings: Settings = Depends(get_settings),
) -> SessionInfo:
    """return metadata about a session's memory state"""
    info = ctx.get_session_info(session_id)
    if not info.get("exists"):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found.")
    return SessionInfo(
        session_id=session_id,
        turn_count=info["turn_count"],
        token_estimate=info["token_estimate"],
        has_summary=info["has_summary"],
        feature_flags={
            "avatar": settings.avatar_enabled,
            "tts": settings.tts_enabled,
            "asr": settings.asr_enabled,
            "rag": settings.rag_enabled,
        },
    )


@router.delete("/session/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(
    session_id: str,
    ctx: ContextManager = Depends(get_context_manager),
) -> None:
    """clear all memory for a session (start fresh)"""
    ctx.delete_session(session_id)
    log.info("session_cleared_via_api", session_id=session_id)


def _sse(event: StreamEvent) -> str:
    """formats a StreamEvent as a SSE string"""
    return f"data: {event.model_dump_json()}\n\n" # double newline coz SSE spec, browsers require it to delimit events
