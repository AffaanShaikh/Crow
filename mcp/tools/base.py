"""
Abstract base class for all MCP tools.

Every tool in this system inherits from BaseTool and implements:
  - 'definition' property  -> the ToolDefinition (schema + description)
  - 'execute()' method     -> the actual implementation
                            

The @tool decorator (in registry.py) handles registration automatically
when a BaseTool subclass is instantiated.

Design principles:
  - Tools are stateless: all state lives in external services (Google API, DB, etc.)
  - Tools validate their own inputs before calling external services
  - Tools return structured data, never raise unhandled exceptions
  - Tools are async - blocking I/O must run in a thread pool
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from typing import Any

from mcp.schemas import ToolCall, ToolDefinition, ToolResult


class BaseTool(ABC):
    """
    Abstract base class for all agent tools. Enforces *Template Method pattern*: 
    the base class defines the algorithm structure, subclasses fill in specific steps.

    We Subclass this and implement 'definition' and 'execute',
    the rest (timing, error catching, result wrapping) is handled here.
    """

    # abstract interface

    @property
    @abstractmethod
    def definition(self) -> ToolDefinition:
        """
        Return the ToolDefinition that describes this tool to the LLM.

        The description must be precise enough that the LLM can decide
        when to use this tool versus another. Vague descriptions lead to
        incorrect tool selection.
        """
        ...

    @abstractmethod
    async def _run(self, **kwargs: Any) -> Any:
        """
        The actual tool implementation.

        Args:   **kwargs matching the tool's parameter schema.
        Returns: Any JSON-serialisable value.
        Raises:  Any exception - caught and wrapped by execute().
        """
        ...

    # concrete execute wrapper

    async def execute(self, call: ToolCall) -> ToolResult:
        """
        Execute the tool call and return a ToolResult.

        Handles:
          - Input validation against the tool's schema
            - arg. validation + coercion (see _validate_arguments docstring)
          - Execution timing
          - Exception catching (always returns ToolResult, never raises) and structured error reporting
          - Never raises - always returns a ToolResult 
        """
        t0 = time.perf_counter()
        try:
            validated_args = self._validate_arguments(call.arguments)
            output = await self._run(**validated_args)
            execution_ms = (time.perf_counter() - t0) * 1000
            return ToolResult(
                call_id=call.id,
                tool_name=call.tool_name,
                success=True,
                output=output,
                execution_ms=round(execution_ms, 2),
            )
        except ToolValidationError as exc:
            execution_ms = (time.perf_counter() - t0) * 1000
            return ToolResult(
                call_id=call.id,
                tool_name=call.tool_name,
                success=False,
                output=None,
                error=f"Invalid arguments: {exc}",
                execution_ms=round(execution_ms, 2),
            )
        except Exception as exc:
            execution_ms = (time.perf_counter() - t0) * 1000
            return ToolResult(
                call_id=call.id,
                tool_name=call.tool_name,
                success=False,
                output=None,
                error=str(exc),
                execution_ms=round(execution_ms, 2),
            )

    # arg. validation

    def _validate_arguments(self, raw_args: dict[str, Any]) -> dict[str, Any]:
        """
        Validates and coerces tool arguments against the tool's parameter schema.

        Coercions applied (in priority order):
          1. Array params as JSON string  -> parsed list
             LLMs often pass ["a","b"] as the string '["a","b"]'
          2. Boolean strings              -> bool ("false" -> False)
          3. Integer/number strings       -> numeric types
          4. Empty string for optional    -> use default (treat as absent)
          5. Enum validation
          6. Required-but-missing guard 
        """
        schema = self.definition.parameters
        validated: dict[str, Any] = {}

        for param_name, prop in schema.properties.items():
            is_required = param_name in schema.required
            raw_value = raw_args.get(param_name, _MISSING)

            # arg. was provided
            if raw_value is not _MISSING:
                value = raw_value

                # coercion 1: array param sent as JSON string
                # e.g. attendees = '["alice@x.com","bob@y.com"]'  (a string)
                # or   attendees = '[]'
                if prop.type == "array" and isinstance(value, str):
                    stripped = value.strip()
                    if stripped.startswith("["):
                        try:
                            value = json.loads(stripped)
                        except json.JSONDecodeError:
                            # treat as a single-element list if it looks like content
                            value = [stripped] if stripped not in ("[]", "") else []
                    elif stripped == "":
                        value = []
                    else:
                        # single value without brackets -> wrap in list
                        value = [stripped]

                # coercion 2: boolean strings
                elif prop.type == "boolean" and isinstance(value, str):
                    value = value.strip().lower() in ("true", "1", "yes")

                # coercion 3: empty string for optional string -> use default,
                # llms pass "" when they have no value for an optional param.
                elif (
                    prop.type == "string"
                    and isinstance(value, str)
                    and value.strip() == ""
                    and not is_required
                ):
                    value = prop.default
                    if value is None:
                        continue # no default -> omit entirely

                else:
                    # coercion 4: numeric strings
                    try:
                        if prop.type == "integer" and not isinstance(value, int):
                            value = int(value)
                        elif prop.type == "number" and not isinstance(value, (int, float)):
                            value = float(value)
                    except (ValueError, TypeError) as exc:
                        raise ToolValidationError(
                            f"Parameter '{param_name}' should be {prop.type}: {exc}"
                        )

                # enum validation
                if prop.enum and value not in prop.enum:
                    raise ToolValidationError(
                        f"Parameter '{param_name}' must be one of {prop.enum}, got '{value}'"
                    )

                validated[param_name] = value

            # arg. required, but is missing, has a default
            elif is_required:
                if prop.default is not None:
                    validated[param_name] = prop.default
                else:
                    raise ToolValidationError(
                        f"Required parameter '{param_name}' is missing"
                    )
            elif prop.default is not None:
                validated[param_name] = prop.default
            # else: optional with no default -> omit

        return validated

    def __repr__(self) -> str:
        return f"<Tool: {self.definition.name}>"


# Sentinel for "argument not provided at all"
class _Missing:
    def __repr__(self): return "<MISSING>"

_MISSING = _Missing()


class ToolValidationError(ValueError):
    """Raised when tool arguments fail validation."""
    pass