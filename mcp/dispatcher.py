"""
The Tools Dispatcher.

The single point through which ALL tool executions flow.
The agent loop calls dispatcher.dispatch(tool_call) - the dispatcher
handles everything else: lookup, timeout, error wrapping, logging.

Separating dispatch from the agent loop means:
  - Timeouts are enforced consistently for every tool
  - All tool execution is logged in one place
  - Can add rate limiting, auth checks, dry-run mode here
    without touching the agent loop or individual tools
"""

from __future__ import annotations

import asyncio
import json
import time

from mcp.registry import ToolRegistry
from mcp.schemas import ToolCall, ToolCallStatus, ToolResult
from utils.logger import get_logger

log = get_logger(__name__)

# hard ceiling: no tool can run longer than this regardless of its own timeout
ABSOLUTE_MAX_TIMEOUT = 120.0


class ToolDispatcher:
    """
    Executes ToolCall objects by looking up the right tool and running it.

    Usage::

        dispatcher = ToolDispatcher(registry)
        result = await dispatcher.dispatch(tool_call)
        print(result.output)
    """

    def __init__(self, registry: ToolRegistry, dry_run: bool = False) -> None:
        """
        Args:
            registry: The tool registry to look up implementations.
            dry_run:  for testing, if True, log tool calls but don't execute them. 
        """
        self.registry = registry
        self.dry_run = dry_run
        log.info("dispatcher_init", dry_run=dry_run)

    async def dispatch(self, call: ToolCall) -> ToolResult:
        """
        Executes a single tool call.

        Steps:-
          1. Look up the tool in the registry
          2. Check it's enabled
          3. Run with timeout
          4. Return ToolResult (never raises)
        """
        call.status = ToolCallStatus.RUNNING
        log.info("tool_dispatch_start",
                 tool=call.tool_name, call_id=call.id,
                 args=_sanitise_args(call.arguments))

        # look up the tool

        tool = self.registry.get_tool(call.tool_name)
        if tool is None:
            log.warning("tool_not_found", name=call.tool_name)
            call.status = ToolCallStatus.ERROR
            return ToolResult(
                call_id=call.id,
                tool_name=call.tool_name,
                success=False,
                output=None,
                error=f"Tool '{call.tool_name}' is not registered. "
                      f"Available: {list(self.registry._tools.keys())}",
            )

        meta = self.registry.get_metadata(call.tool_name)
        if meta and not meta.is_enabled:
            call.status = ToolCallStatus.ERROR
            return ToolResult(
                call_id=call.id,
                tool_name=call.tool_name,
                success=False,
                output=None,
                error=f"Tool '{call.tool_name}' is currently disabled.",
            )

        # dry run tool testing
        if self.dry_run:
            log.info("tool_dry_run", tool=call.tool_name, args=call.arguments)
            call.status = ToolCallStatus.SUCCESS
            return ToolResult(
                call_id=call.id,
                tool_name=call.tool_name,
                success=True,
                output={"dry_run": True, "would_have_called": call.tool_name,
                        "with_args": call.arguments},
            )

        # execute w/ timeout
        timeout = min(
            meta.timeout_seconds if meta else 30.0,
            ABSOLUTE_MAX_TIMEOUT,
        )

        try:
            result = await asyncio.wait_for(tool.execute(call), timeout=timeout)
            call.status = ToolCallStatus.SUCCESS if result.success else ToolCallStatus.ERROR

            log.info(
                "tool_dispatch_done",
                tool=call.tool_name,
                success=result.success,
                execution_ms=result.execution_ms,
                error=result.error,
            )
            return result

        except asyncio.TimeoutError:
            log.error("tool_timeout", tool=call.tool_name, timeout_s=timeout)
            call.status = ToolCallStatus.TIMEOUT
            return ToolResult(
                call_id=call.id,
                tool_name=call.tool_name,
                success=False,
                output=None,
                error=f"Tool timed out after {timeout}s.",
                execution_ms=timeout * 1000,
            )
        except Exception as exc:
            log.exception("tool_dispatch_unexpected_error",
                          tool=call.tool_name, error=str(exc))
            call.status = ToolCallStatus.ERROR
            return ToolResult(
                call_id=call.id,
                tool_name=call.tool_name,
                success=False,
                output=None,
                error=f"Unexpected dispatcher error: {exc}",
            )

    async def dispatch_many(self, calls: list[ToolCall]) -> list[ToolResult]:
        """
        Executes multiple tool calls concurrently.

        Use when the llm requests several independent tool calls in the same
        response. Parallel execution dramatically reduces total latency when
        tools involve network I/O (like multiple calendar queries for example).
        """
        if not calls:
            return []
        tasks = [self.dispatch(call) for call in calls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        final: list[ToolResult] = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                final.append(ToolResult(
                    call_id=calls[i].id,
                    tool_name=calls[i].tool_name,
                    success=False,
                    output=None,
                    error=str(r),
                ))
            else:
                final.append(r)
        return final


# helper func.(s)

def _sanitise_args(args: dict) -> dict:
    """Remove potentially sensitive values from args before logging."""
    REDACT = {"password", "token", "secret", "key", "credential"}
    return {
        k: "***" if any(s in k.lower() for s in REDACT) else v
        for k, v in args.items()
    }


def parse_tool_calls_from_response(response_message: dict) -> list[ToolCall]:
    """
    Parses tool calls from an OpenAI-format chat completion message.

    Handles both the standard 'tool_calls' array format (used by
    llama.cpp with tool-calling capable models) and returns an empty list for plain text responses.
    """
    tool_calls_raw = response_message.get("tool_calls") or []
    calls: list[ToolCall] = []

    for tc in tool_calls_raw:
        try:
            fn = tc.get("function", {})
            raw_args = fn.get("arguments", "{}")
            # since llms return arguments as a string or sometimes as a dict
            if isinstance(raw_args, str):
                arguments = json.loads(raw_args)
            else:
                arguments = raw_args

            calls.append(ToolCall(
                id=tc.get("id", f"call_{len(calls)}"),
                tool_name=fn.get("name", ""),
                arguments=arguments,
            ))
        except (json.JSONDecodeError, KeyError) as exc:
            log.warning("tool_call_parse_error", raw=tc, error=str(exc))

    return calls
