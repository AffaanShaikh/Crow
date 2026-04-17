"""
Agent API route.

POST /agent/stream - SSE stream from the full agent loop
GET  /agent/tools  - list registered tools and their status

This route ALWAYS calls agent.run_streaming().
The ToolRouter inside the agent loop is the single classification authority:

  Path A (router returns [])       -> stream directly through persona
  Path B (router returns [tools])  -> tool orchestration -> streaming synthesis

SSE event types (superset of chat events):
  {"type": "delta",       "content": "token..."}                                    text token (both paths)
  {"type": "tool_start",  "tool_name": "list_calendar_events", "call_id": "..."}    tool invoked  (Path B)
  {"type": "tool_done",   "tool_name": "...", "success": true, "execution_ms": 120} tool result   (Path B)
  {"type": "thinking",    "content": "Let me check your calendar..."}               LLM reasoning (Path B)
  {"type": "done",        "metadata": {...}}                                        stream complete
  {"type": "error",       "error": "..."}                                           error

To add further tools: register tools in registry.py + patterns in router.py.
Nothing in this file ever needs to change.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator

import structlog
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

# from config import get_settings, Settings
# from llm.client import get_llm_client
from llm.prompt_builder import build_messages
from mcp.agent_loop import AgentLoop, get_agent_loop
from mcp.registry import ToolRegistry, get_registry
from mcp.schemas import AgentStepType
from memory.context_manager import ContextManager, get_context_manager
from models.schemas import ChatRequest, StreamEvent, StreamEventType
from utils.logger import get_logger

log = get_logger(__name__)
router = APIRouter(prefix="/agent", tags=["agent"])


@router.post("/stream")
async def agent_stream(
    req: ChatRequest,
    request: Request,
    agent: AgentLoop = Depends(get_agent_loop),
    ctx: ContextManager = Depends(get_context_manager),
    # registry: ToolRegistry = Depends(get_registry),
    # settings: Settings = Depends(get_settings),
) -> StreamingResponse:
    """
    Full agent loop with streaming SSE output.

    Events sent to the frontend:
      - tool_start / tool_done  -> show "Checking calendar.." indicators
      - thinking                -> optional reasoning text between tool calls
      - delta                   -> final answer tokens (streamed character by character)
      - done                    -> stream complete

    The ToolRouter inside agent.run_streaming() classifies the message and
    routes to Path A (direct persona stream) or Path B (tool use + synthesis).
    The frontend has no conditional logic, it always connects here.
    """
    request_id = str(uuid.uuid4())[:8]
    structlog.contextvars.bind_contextvars(
        request_id=request_id,
        session_id=req.session_id,
    )
    log.info("agent_stream_start", message_preview=req.message[:80])

    async def event_generator() -> AsyncIterator[str]:
        t0 = time.perf_counter()
        # full_response_parts: list[str] = []
        full_response_text: str = ""

        try:
            history, summary = await ctx.get_context(req.session_id)
            log.info("history&summary agent_stream", history=history, summary=summary)
            messages = build_messages(
                user_message=req.message,
                history_turns=history,
                summary=summary,
            )

            async for step in agent.run_streaming(messages, req.session_id):

                if await request.is_disconnected():
                    log.info("client_disconnected_mid_agent")
                    break

                if step.type == AgentStepType.DELTA and step.content:
                    # real streaming token - forward immediately for that sweet typewriter effect
                    yield f"data: {StreamEvent(type=StreamEventType.DELTA, content=step.content).model_dump_json()}\n\n"

                elif step.type == AgentStepType.THINKING and step.content:
                    # reasoning text shown as dimmed prefix in UI
                    yield _sse({"type": "thinking", "content": step.content})

                elif step.type == AgentStepType.TOOL_CALL and step.tool_call:
                    yield _sse({
                        "type": "tool_start",
                        "tool_name": step.tool_call.tool_name,
                        "call_id": step.tool_call.id,
                        "arguments": step.tool_call.arguments,
                    })

                elif step.type == AgentStepType.TOOL_RESULT and step.tool_result:
                    yield _sse({
                        "type": "tool_done",
                        "tool_name": step.tool_result.tool_name,
                        "call_id": step.tool_result.call_id,
                        "success": step.tool_result.success,
                        "execution_ms": step.tool_result.execution_ms,
                        "error": step.tool_result.error,
                    })

                elif step.type == AgentStepType.FINAL and step.content:
                    # # stream the final answer token by token for the typewriter effect
                    # # (Agent loop returns full string - simulate streaming with chunks)
                    # answer = step.content
                    # full_response_parts.append(answer)

                    # # Chunk into ~3-char pieces to simulate streaming
                    # chunk_size = 3
                    # for i in range(0, len(answer), chunk_size):
                    #     chunk = answer[i:i + chunk_size]
                    #     event = StreamEvent(
                    #         type=StreamEventType.DELTA,
                    #         content=chunk,
                    #     )
                    #     yield f"data: {event.model_dump_json()}\n\n"

                    # for memory only - every character already delivered via DELTA.
                    full_response_text = step.content

            # # save complete exchange to memory
            # complete_response = "".join(full_response_parts)
            # if complete_response:
            #     await ctx.add_turn(
            #         session_id=req.session_id,
            #         user_message=req.message,
            #         assistant_response=complete_response,
            #     )
            if full_response_text:
                await ctx.add_turn(
                    session_id=req.session_id,
                    user_message=req.message,
                    assistant_response=full_response_text,
                )

            elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
            done_event = StreamEvent(
                type=StreamEventType.DONE,
                session_id=req.session_id,
                metadata={"elapsed_ms": elapsed_ms, "request_id": request_id},
            )
            yield f"data: {done_event.model_dump_json()}\n\n"
            log.info("agent_stream_done", elapsed_ms=elapsed_ms, chars=len(full_response_text))

        except Exception as exc:
            log.exception("agent_stream_error", error=str(exc))
            error_event = StreamEvent(
                type=StreamEventType.ERROR,
                error="The agent encountered an error. Please try again.",
            )
            yield f"data: {error_event.model_dump_json()}\n\n"
        finally:
            structlog.contextvars.clear_contextvars()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@router.get("/tools")
async def list_tools(registry: ToolRegistry = Depends(get_registry)) -> dict:
    """Return all registered tools with their status and descriptions."""
    return {
        "tools": registry.list_tools(),
        "total": len(registry),
    }


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"
