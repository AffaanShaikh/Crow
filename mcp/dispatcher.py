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

from datetime import datetime, timedelta, timezone
import re
import calendar 

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
          2. Check if it's enabled
          3. Run with timeout
          4. Return ToolResult (never raises)
        """
        call.status = ToolCallStatus.RUNNING
        log.info("tool_dispatch_start",
                 tool=call.tool_name, call_id=call.id,
                 args=_sanitise_args3(call.arguments), unsanatizedargs=call.arguments)

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
def _sanitise_args2(args: dict) -> dict:
    """
    Sanitize args while normalizing common LLM argument issues.

    - Redacts sensitive keys/values
    - Converts common datetime tokens ('now', 'today', 'tomorrow') to ISO-8601
    - Coerces obvious numeric/boolean strings
    - Returns the same dict structure expected by existing logging
    """

    REDACT_KEYS = {"password", "token", "secret", "key", "credential"}
    DATETIME_KEYS = {"time", "date", "start", "end"}
    TOKEN_RE = re.compile(r"^[A-Za-z0-9-_]+\.[A-Za-z0-9-_]+\.[A-Za-z0-9-_]+$")

    def redact_value(k, v):
        if any(s in k.lower() for s in REDACT_KEYS):
            return "***"
        if isinstance(v, str) and (TOKEN_RE.match(v) or len(v) > 120):
            return "***"
        return v

    def normalize_value(k, v):
        if not isinstance(v, str):
            return v

        vs = v.strip()

        # datetime normalization
        if any(h in k.lower() for h in DATETIME_KEYS):
            token = vs.lower()

            now = datetime.now(timezone.utc)

            if token == "now":
                return now.isoformat()

            if token == "today":
                dt = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
                return dt.isoformat()

            if token == "tomorrow":
                d = now.date() + timedelta(days=1)
                dt = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
                return dt.isoformat()

        # integer coercion
        if re.fullmatch(r"-?\d+", vs):
            try:
                return int(vs)
            except Exception:
                pass

        # float coercion
        if re.fullmatch(r"-?\d+\.\d+", vs):
            try:
                return float(vs)
            except Exception:
                pass

        # boolean coercion
        if vs.lower() in {"true", "false", "yes", "no", "1", "0"}:
            return vs.lower() in {"true", "yes", "1"}

        return v

    sanitized = {}

    for k, v in (args or {}).items():
        normalized = normalize_value(k, v)
        sanitized[k] = redact_value(k, normalized)

    return sanitized

def _sanitise_args3(args: dict) -> dict:
    """
    Sanitizes args while normalizing common LLM argument issues.

    - Redacts sensitive keys/values
    - Converts common datetime tokens ('now', 'today', 'tomorrow', 'this week', 'in 3 days', '3 days ago', etc.) to ISO-8601
    - Coerces obvious numeric/boolean strings
    - Returns the same dict structure expected by existing logging
    """

    REDACT_KEYS = {"password", "token", "secret", "key", "credential"}
    DATETIME_KEYS = {"time", "date", "start", "end"}
    TOKEN_RE = re.compile(r"^[A-Za-z0-9-_]+\.[A-Za-z0-9-_]+\.[A-Za-z0-9-_]+$")

    def redact_value(k, v):
        if any(s in k.lower() for s in REDACT_KEYS):
            return "***"
        if isinstance(v, str) and (TOKEN_RE.match(v) or len(v) > 120):
            return "***"
        return v

    def add_months(dt: datetime, months: int) -> datetime:
        # safely add months (keeps day within valid range)
        month = dt.month - 1 + months
        year = dt.year + month // 12
        month = month % 12 + 1
        day = min(dt.day, calendar.monthrange(year, month)[1])
        return dt.replace(year=year, month=month, day=day)

    def parse_relative_datetime(token: str, now: datetime):
        """
        Try parsing human-friendly relative datetime tokens.
        Returns a datetime or None if not parsed.
        """
        t = token.strip().lower()

        # exact tokens
        if t == "now":
            return now
        if t == "today":
            return datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
        if t == "tomorrow":
            d = now.date() + timedelta(days=1)
            return datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
        if t == "yesterday":
            d = now.date() - timedelta(days=1)
            return datetime(d.year, d.month, d.day, tzinfo=timezone.utc)

        # this/next/last week/month/year
        if t in {"this week", "next week", "last week"}:
            # start of week = Monday
            start_of_week = datetime(now.year, now.month, now.day, tzinfo=timezone.utc) - timedelta(days=now.weekday())
            if t == "this week":
                return start_of_week
            if t == "next week":
                return start_of_week + timedelta(weeks=1)
            return start_of_week - timedelta(weeks=1)

        if t in {"this month", "next month", "last month"}:
            first_of_month = datetime(now.year, now.month, 1, tzinfo=timezone.utc)
            if t == "this month":
                return first_of_month
            if t == "next month":
                return add_months(first_of_month, 1)
            return add_months(first_of_month, -1)

        if t in {"this year", "next year", "last year"}:
            first_of_year = datetime(now.year, 1, 1, tzinfo=timezone.utc)
            if t == "this year":
                return first_of_year
            if t == "next year":
                return datetime(now.year + 1, 1, 1, tzinfo=timezone.utc)
            return datetime(now.year - 1, 1, 1, tzinfo=timezone.utc)

        # patterns like "in 3 days", "3 days ago", "next 2 weeks", "last 2 months"
        m = re.match(r'^(?:in\s+)?(\d+)\s+(hour|hours|minute|minutes|day|days|week|weeks|month|months|year|years)(?:\s+ago)?$', t)
        if m:
            n = int(m.group(1))
            unit = m.group(2)
            # determine sign: "in X" positive, "X ... ago" negative (if "ago" exists), or if neither assume positive
            sign = 1
            if t.endswith("ago"):
                sign = -1
            elif t.startswith("in "):
                sign = 1
            # convert
            if unit.startswith("minute"):
                return now + sign * timedelta(minutes=n)
            if unit.startswith("hour"):
                return now + sign * timedelta(hours=n)
            if unit.startswith("day"):
                target = now + sign * timedelta(days=n)
                return datetime(target.year, target.month, target.day, tzinfo=timezone.utc)
            if unit.startswith("week"):
                target = now + sign * timedelta(weeks=n)
                start_of_week = datetime(target.year, target.month, target.day, tzinfo=timezone.utc) - timedelta(days=target.weekday())
                return start_of_week
            if unit.startswith("month"):
                # add months preserving day where possible; for "in N months" we return the same day/time shifted
                target = add_months(now, sign * n)
                return datetime(target.year, target.month, target.day, tzinfo=timezone.utc)
            if unit.startswith("year"):
                try:
                    return datetime(now.year + sign * n, now.month, now.day, tzinfo=timezone.utc)
                except Exception:
                    # fallback to first of year
                    return datetime(now.year + sign * n, 1, 1, tzinfo=timezone.utc)

        # patterns like "next week", "last month", "next 2 weeks" (covered above partly)
        m2 = re.match(r'^(next|last)\s+(\d+)?\s*(hour|hours|minute|minutes|day|days|week|weeks|month|months|year|years)?$', t)
        if m2:
            when = m2.group(1)
            num = int(m2.group(2)) if m2.group(2) else 1
            unit = m2.group(3) or "week"  # default to week if not given (e.g. "next")
            sign = 1 if when == "next" else -1
            if unit.startswith("minute"):
                return now + sign * timedelta(minutes=num)
            if unit.startswith("hour"):
                return now + sign * timedelta(hours=num)
            if unit.startswith("day"):
                target = now + sign * timedelta(days=num)
                return datetime(target.year, target.month, target.day, tzinfo=timezone.utc)
            if unit.startswith("week"):
                target = now + sign * timedelta(weeks=num)
                start_of_week = datetime(target.year, target.month, target.day, tzinfo=timezone.utc) - timedelta(days=target.weekday())
                return start_of_week
            if unit.startswith("month"):
                target = add_months(now, sign * num)
                return datetime(target.year, target.month, target.day, tzinfo=timezone.utc)
            if unit.startswith("year"):
                try:
                    return datetime(now.year + sign * num, now.month, now.day, tzinfo=timezone.utc)
                except Exception:
                    return datetime(now.year + sign * num, 1, 1, tzinfo=timezone.utc)

        # not parsed
        return None

    def normalize_value(k, v):
        if not isinstance(v, str):
            return v

        vs = v.strip()

        # datetime normalization
        if any(h in k.lower() for h in DATETIME_KEYS):
            token = vs.lower()

            now = datetime.now(timezone.utc)

            # try extended parser for human-friendly tokens
            parsed = parse_relative_datetime(token, now)
            if parsed is not None:
                return parsed.isoformat()

            # fallback to the previous exact tokens logic (kept for compatibility)
            if token == "now":
                return now.isoformat()

            if token == "today":
                dt = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
                return dt.isoformat()

            if token == "tomorrow":
                d = now.date() + timedelta(days=1)
                dt = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
                return dt.isoformat()

        # integer coercion
        if re.fullmatch(r"-?\d+", vs):
            try:
                return int(vs)
            except Exception:
                pass

        # float coercion
        if re.fullmatch(r"-?\d+\.\d+", vs):
            try:
                return float(vs)
            except Exception:
                pass

        # boolean coercion
        if vs.lower() in {"true", "false", "yes", "no", "1", "0"}:
            return vs.lower() in {"true", "yes", "1"}

        return v

    sanitized = {}

    for k, v in (args or {}).items():
        normalized = normalize_value(k, v)
        sanitized[k] = redact_value(k, normalized)

    return sanitized

def parse_tool_calls_from_response(response_message: dict) -> list[ToolCall]:
    """
    Parses tool calls from an OpenAI-format chat completion message.

    Handles both the standard 'tool_calls' array format (used by llama.cpp with tool-calling capable models) 
    and returns an empty list for plain text responses.
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
