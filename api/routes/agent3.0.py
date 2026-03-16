"""
Agent API route.

POST /agent/stream — Intent-routed SSE stream
GET  /agent/tools  — list registered tools

ROUTING LOGIC (the fix for spurious tool calls):
  Every message is classified before touching the agent loop.
  Only messages that match calendar intent keywords reach the agent.
  Everything else goes straight to the plain LLM stream — no tools,
  no hallucinated IDs, no 10-second waits for conversational replies.

  chat path:  message → LLM stream → delta events
  agent path: message → agent loop → tool_start/tool_done/delta events

SSE event types:
  {"type": "delta",      "content": "token..."}          — text token
  {"type": "tool_start", "tool_name": "...", "call_id": "..."} — tool invoked
  {"type": "tool_done",  "tool_name": "...", "success": true}  — tool finished
  {"type": "thinking",   "content": "..."}               — LLM reasoning text
  {"type": "done",       "metadata": {...}}               — stream complete
  {"type": "error",      "error": "..."}                  — something went wrong
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator

import structlog
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from config import get_settings, Settings
from llm.client import LLMClient, get_llm_client
from llm.prompt_builder import build_messages, build_agent_system_prompt
from mcp.agent_loop import AgentLoop, get_agent_loop
from mcp.registry import ToolRegistry, get_registry
from mcp.schemas import AgentStepType
from memory.context_manager import ContextManager, get_context_manager
from models.schemas import ChatRequest, StreamEvent, StreamEventType
from utils.logger import get_logger

log = get_logger(__name__)
router = APIRouter(prefix="/agent", tags=["agent"])


# ── Intent classification ─────────────────────────────────────────────────────
# Deterministic keyword matching — zero cost, zero latency, zero hallucination.
# A message must contain at least one of these to enter the agent loop.
# Keep this list conservative: false negatives (missing a tool request) are
# recoverable (user rephrases); false positives (tool on casual chat) are not.

_CALENDAR_KEYWORDS = frozenset({
    # Direct noun references
    "calendar", "event", "events", "schedule", "meeting", "meetings",
    "appointment", "appointments", "booking",
    # Action verbs applied to scheduling
    "remind", "reschedule", "cancel meeting", "cancel event",
    "book a", "set up a meeting", "set up a call", "arrange a",
    # CRUD phrasing
    "add event", "create event", "make event", "new event",
    "delete event", "remove event", "update event", "change event",
    "move meeting", "move the meeting", "edit event",
    # Query phrasing
    "what's on", "what do i have", "what have i got",
    "am i free", "am i busy", "any meetings", "any events",
    "free time", "free slot", "available",
    "my day", "my week", "my schedule", "my calendar",
    "upcoming", "next meeting", "next event", "last meeting",
    # Listing
    "list events", "list my", "show events", "show my calendar",
    "what is on", "what's scheduled",
})


def _needs_tools(message: str) -> bool:
    """
    Return True if the message contains explicit calendar/scheduling intent.

    This is the gatekeeper for the agent loop. The design is intentionally
    conservative — only unambiguous scheduling language passes through.
    General conversation, questions about the assistant, knowledge questions,
    and anything else goes straight to plain LLM streaming.
    """
    lower = message.lower()
    return any(kw in lower for kw in _CALENDAR_KEYWORDS)


# ── Main endpoint ─────────────────────────────────────────────────────────────

@router.post("/stream")
async def agent_stream(
    req: ChatRequest,
    request: Request,
    llm: LLMClient = Depends(get_llm_client),
    agent: AgentLoop = Depends(get_agent_loop),
    ctx: ContextManager = Depends(get_context_manager),
    registry: ToolRegistry = Depends(get_registry),
    settings: Settings = Depends(get_settings),
) -> StreamingResponse:
    """
    Intent-routed streaming endpoint.

    Classifies the message first, then takes one of two code paths:
      "chat"  → plain LLM stream, no tools (fast, no hallucination risk)
      "agent" → full tool-use loop (only when calendar keywords detected)
    """
    request_id = str(uuid.uuid4())[:8]
    structlog.contextvars.bind_contextvars(
        request_id=request_id,
        session_id=req.session_id,
    )

    use_tools = _needs_tools(req.message)
    log.info(
        "agent_stream_start",
        message_preview=req.message[:80],
        route="agent" if use_tools else "chat",
    )

    async def event_generator() -> AsyncIterator[str]:
        t0 = time.perf_counter()
        full_response: list[str] = []

        try:
            history, summary = await ctx.get_context(req.session_id)

            # ── CHAT PATH: plain streaming, no tools ──────────────────────
            if not use_tools:
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
                    if await request.is_disconnected():
                        break
                    full_response.append(token)
                    yield f"data: {StreamEvent(type=StreamEventType.DELTA, content=token).model_dump_json()}\n\n"

            # ── AGENT PATH: full tool-use loop ────────────────────────────
            else:
                # Build messages with the tool-aware system prompt variant
                messages = build_messages(
                    user_message=req.message,
                    history_turns=history,
                    summary=summary,
                    mode="agent",   # appends tool use policy to system prompt
                )

                async for step in agent.run_streaming(messages, req.session_id):
                    if await request.is_disconnected():
                        log.info("client_disconnected_mid_agent")
                        break

                    if step.type == AgentStepType.THINKING and step.content:
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
                        answer = step.content
                        full_response.append(answer)
                        # Chunk final answer into small pieces for typewriter effect
                        for i in range(0, len(answer), 4):
                            chunk = answer[i:i + 4]
                            yield f"data: {StreamEvent(type=StreamEventType.DELTA, content=chunk).model_dump_json()}\n\n"

            # ── Persist turn ──────────────────────────────────────────────
            complete = "".join(full_response)
            if complete:
                await ctx.add_turn(
                    session_id=req.session_id,
                    user_message=req.message,
                    assistant_response=complete,
                )

            elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
            yield f"data: {StreamEvent(type=StreamEventType.DONE, session_id=req.session_id, metadata={'elapsed_ms': elapsed_ms, 'request_id': request_id, 'route': 'agent' if use_tools else 'chat'}).model_dump_json()}\n\n"
            log.info("agent_stream_done", elapsed_ms=elapsed_ms, route="agent" if use_tools else "chat")

        except Exception as exc:
            log.exception("agent_stream_error", error=str(exc))
            yield f"data: {StreamEvent(type=StreamEventType.ERROR, error='An error occurred. Please try again.').model_dump_json()}\n\n"
        finally:
            structlog.contextvars.clear_contextvars()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


@router.get("/tools")
async def list_tools(registry: ToolRegistry = Depends(get_registry)) -> dict:
    """Return all registered tools with their status and descriptions."""
    return {"tools": registry.list_tools(), "total": len(registry)}


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"