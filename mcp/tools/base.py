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
          - Execution timing
          - Exception catching and structured error reporting
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

    # validation

    def _validate_arguments(self, raw_args: dict[str, Any]) -> dict[str, Any]:
        """
        Check that required parameters are present and types are correct.
        Raises ToolValidationError on failure.
        """
        schema = self.definition.parameters
        validated: dict[str, Any] = {}

        for param_name, prop in schema.properties.items():
            if param_name in raw_args:
                value = raw_args[param_name]
                # basic type coercion
                try:
                    if prop.type == "integer" and not isinstance(value, int):
                        value = int(value)
                    elif prop.type == "number" and not isinstance(value, float):
                        value = float(value)
                    elif prop.type == "boolean" and not isinstance(value, bool):
                        value = str(value).lower() in ("true", "1", "yes")
                except (ValueError, TypeError) as exc:
                    raise ToolValidationError(
                        f"Parameter '{param_name}' should be {prop.type}: {exc}"
                    )
                # Enum validation
                if prop.enum and value not in prop.enum:
                    raise ToolValidationError(
                        f"Parameter '{param_name}' must be one of {prop.enum}, got '{value}'"
                    )
                validated[param_name] = value
            elif param_name in schema.required:
                if prop.default is not None:
                    validated[param_name] = prop.default
                else:
                    raise ToolValidationError(f"Required parameter '{param_name}' is missing")
            elif prop.default is not None:
                validated[param_name] = prop.default

        return validated

    def __repr__(self) -> str:
        return f"<Tool: {self.definition.name}>"


class ToolValidationError(ValueError):
    """Raised when tool arguments fail validation."""
    pass
