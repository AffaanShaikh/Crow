"""
The Agent Loop.  

The loop can run multiple cycles ("turns" internally) when the llm calls tools in sequence.

If the model returns has_tool_calls=False despite tools being available,
this file will log a WARNING with the model name and a fix suggestion.

Loop behaviour:
  1. Build messages + inject agent system addendum
  2. Call llm with tools schema
  3. Parse tool_calls from response
  4. If none -> yield FINAL, return
  5. Dispatch tools (parallel), inject results
  6. Repeat until no tool_calls or max_iterations

The loop has two safety limits:-
  - max_iterations: prevents runaway loops (default: 8)
  - The llm choosing not to call tools exits immediately

Streaming support:
  The loop yields AgentStep events as they happen so the frontend can
  show live tool-use indicators (ex: "Checking your calendar..")
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

from config import get_settings
from mcp.dispatcher import ToolDispatcher, parse_tool_calls_from_response
from mcp.registry import ToolRegistry
from mcp.schemas import AgentResponse, AgentStep, AgentStepType, ToolCall
from models.schemas import Message, Role
from utils.logger import get_logger

log = get_logger(__name__)
settings = get_settings()

MAX_ITERATIONS = 8              # hard cap on tool-call cycles per user message
MAX_TOOL_CALLS_PER_STEP = 5     # max parallel tool calls in one iteration

# injected as the LAST system message so it overrides persona tone for tool use,
# keeps the persona system prompt intact while sorta neccessitating tool-calling instructions
TOOL_USE_ADDENDUM = """\

## Tool Use Instructions
You have access to tools listed in the 'tools' parameter. Use them if and when necessary."""
# When asked about calendar, schedule, events, or any action you can
# fulfil with a tool - you MUST call the appropriate tool rather than guessing or
# describing what you would do.
# Do not fabricate calendar data. If you need real data, call the tool.
# After receiving tool results, synthesise them into a natural response in your voice.

class AgentLoop:
    """
    Orchestrates multi-step tool-use reasoning. (over an OpenAI-compatible llm)

    Works with any OpenAI-compatible llm client. Requires a model that
    supports function calling.

    Usage::

        loop = AgentLoop(llm_client, dispatcher)

        # Non-streaming (waits for full response)
        response = await loop.run(messages, tools)

        # Streaming (yields steps as they happen)
        async for step in loop.run_streaming(messages, tools):
            if step.type == AgentStepType.TOOL_CALL:
                print(f"Calling {step.tool_call.tool_name}...")
            elif step.type == AgentStepType.FINAL:
                print(step.content)
    """

    def __init__(
        self,
        llm_client, # LLMClient - passed in to avoid circular import
        dispatcher: ToolDispatcher,
        registry: ToolRegistry,
        max_iterations: int = MAX_ITERATIONS,
    ) -> None:
        self.llm = llm_client
        self.dispatcher = dispatcher
        self.registry = registry
        self.max_iterations = max_iterations

    # non-streaming run

    async def run(
        self,
        messages: list[Message],
        session_id: str,
    ) -> AgentResponse:
        """
        Run the full agent loop and return a complete AgentResponse.
        Blocks until the LLM produces a final answer with no more tool calls.
        """
        steps: list[AgentStep] = []
        final_text: list[str] = []

        async for step in self.run_streaming(messages, session_id):
            steps.append(step)
            if step.type == AgentStepType.FINAL and step.content:
                final_text.append(step.content)

        return AgentResponse(
            final_answer="".join(final_text) or "Unable to complete this request.",
            steps=steps,
            total_tool_calls=sum(1 for s in steps if s.type == AgentStepType.TOOL_CALL),
            session_id=session_id,
        )

    # streaming run (primary)

    async def run_streaming(
        self,
        messages: list[Message],
        session_id: str,
    ) -> AsyncIterator[AgentStep]:
        """
        Streaming agent loop. Yields AgentStep events as they happen.

        The frontend subscribes to these to show live reasoning:
          - THINKING        -> LLM is producing text before a tool call
          - TOOL_CALL       -> LLM requested a tool
          - TOOL_RESULT     -> tool executed, result available
          - FINAL           -> LLM's final answer, stream this token-by-token

        Converts Message objects -> dicts, injects tool-use addendum into
        the system message, then loops until no more tool calls.
        """
        # working_messages : accumulates the full conversation across iterations,
        # builds the working message list, we append to this each iteration
        
        # serialise + tool addendum injected into the system message
        working_messages = _prepare_messages(messages)
        tools_schema = self.registry.get_openai_tools()

        log.info(
            "agent_loop_start",
            session_id=session_id,
            available_tools=len(tools_schema),
            message_count=len(working_messages),
            model=settings.llm_model_name,
        )

        for iteration in range(self.max_iterations):
            log.debug("agent_iteration", iteration=iteration, session_id=session_id)

            # llm call
            response_message = await self._call_llm_with_tools(
                working_messages, tools_schema
            )

            text_content: str = response_message.get("content") or ""
            tool_calls_raw: list = response_message.get("tool_calls") or []

            # detect unsupported model, if tools were sent but none called on the first iteration,
            # and the response is long (hallucinated answer), we warn
            if (
                iteration == 0
                and not tool_calls_raw
                and tools_schema
                and len(text_content) > 200
            ):
                log.warning(
                    "model_ignored_tools",
                    model=settings.llm_model_name,
                    response_chars=len(text_content),
                    hint=(
                        f"'{settings.llm_model_name}' returned a plain text answer despite "
                        "tools being available. This model likely does not support function "
                        "calling. Switch to: llama3.2, qwen2.5, mistral-nemo, "
                        "or firefunction-v2. In Ollama: 'ollama pull llama3.2'"
                    ),
                )

            # no tool calls -> final answer
            if not tool_calls_raw:
                log.info(
                    "agent_loop_done",
                    iteration=iteration,
                    session_id=session_id,
                    answer_chars=len(text_content),
                    used_tools=(iteration > 0),
                )
                yield AgentStep(type=AgentStepType.FINAL, content=text_content)
                return

            # if tools invoked, the pre-tool thinking text
            if text_content.strip():
                yield AgentStep(type=AgentStepType.THINKING, content=text_content)

            # parse, cap and emit tool calls
            calls = parse_tool_calls_from_response(response_message)
            calls = calls[:MAX_TOOL_CALLS_PER_STEP] # safety cap
            for call in calls:
                yield AgentStep(type=AgentStepType.TOOL_CALL, tool_call=call)

            # append llm's message (w/ tool_calls) to history
            working_messages.append(response_message)

            # execute tools, in parallel if multiple
            results = await self.dispatcher.dispatch_many(calls)

            # results (as tool-role messages) injected back into message history
            # OpenAI's tool result format: role=tool, one message per result
            for result in results:
                yield AgentStep(type=AgentStepType.TOOL_RESULT, tool_result=result)
                working_messages.append({
                    "role": "tool",
                    "tool_call_id": result.call_id,
                    "content": result.to_message_content(),
                })

            # loop continues, llm sees tool results and decides next action

        # max iterations hit
        log.warning("agent_max_iterations", session_id=session_id, limit=self.max_iterations)
        yield AgentStep(
            type=AgentStepType.FINAL,
            content=(
                "Maximum number of steps for this request has been hit. "
                "Here's what is found so far: " +
                _summarise_tool_results(working_messages)
                #_summarise_results(working_messages)
            ),
        )

    # llm call (raw OpenAI-compatible api)

    async def _call_llm_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
    ) -> dict[str, Any]:
        """
        Calls the LLM with tool schemas and return the raw response message dict.

        Why we bypass LLMClient.complete() here:
          - We need to pass 'tools' and 'tool_choice' parameters
          - We need to inject role=tool messages into the history
          - We need the raw message dict (including tool_calls field),
            not just the text content

        The 'tools' list may be empty if no tools are registered or all
        are disabled. In that case we skip the tools parameter entirely -
        some backends error on an empty tools list.
        """
        kwargs: dict[str, Any] = {
            "model": settings.llm_model_name,   # direct setting, no hack
            "messages": messages,               # use full history including tool results
            "max_tokens": 1024,
            "temperature": 0.1,                 # lower = more deterministic tool selection
        }

        # only pass tools params when we actually have tools
        # coz empty list causes errors on some backends
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"      # let the model decide when to call tools

        log.debug(
            "agent_llm_call",
            message_count=len(messages),
            tool_count=len(tools),
        )

        response = await self.llm._client.chat.completions.create(**kwargs)
        message = response.choices[0].message

        # normalise SDK response object -> plain dict for consistent handling
        result: dict[str, Any] = {
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": [],
        }

        if message.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ]

        log.debug(
            "agent_llm_response",
            has_tool_calls=bool(result["tool_calls"]),
            tool_names=[tc["function"]["name"] for tc in result["tool_calls"]],
            content_chars=len(result["content"]),
        )

        return result

# helper func.(s)

def _prepare_messages(messages: list[Message]) -> list[dict]:
    """
    Converts Message objects -> plain dicts AND inject TOOL_USE_ADDENDUM
    into the system message content.

    Why inject here rather than in prompt_builder?
      coz the prompt_builder builds persona prompts for plain chat too.
      The tool-use addendum is agent-specific - it should only appear
      when the agent loop is running, not in regular chat responses.
    """
    result: list[dict] = []
    for m in messages:
        d = {"role": m.role, "content": m.content}
        if m.role == "system":
            d["content"] = m.content + TOOL_USE_ADDENDUM
        result.append(d)
    return result

def _summarise_tool_results(messages: list[dict]) -> str:
    """Extracts tool result content from message history for max-iterations fallback."""
    results = [
        m["content"][:300]
        for m in messages
        if m.get("role") == "tool" and m.get("content")
    ]
    return " | ".join(results) if results else "No tool results obtained."

# singletons

_agent_loop: AgentLoop | None = None


def get_agent_loop() -> AgentLoop:
    if _agent_loop is None:
        raise RuntimeError("AgentLoop not initialised. Check MCP_ENABLED=true in .env")
    return _agent_loop


def init_agent_loop(llm_client, dispatcher: ToolDispatcher, registry: ToolRegistry) -> AgentLoop:
    global _agent_loop
    _agent_loop = AgentLoop(llm_client, dispatcher, registry)
    return _agent_loop
