"""
Tool Registry.

Central catalogue of all available tools. Tools register themselves here
and the agent loop queries the registry to:
  - Get tool schemas to send to the LLM
  - Look up which Python class to call when the LLM requests a tool

Registration happens automatically when you add a tool class to the
'_TOOL_CLASSES' (ToolRegistry) list at the bottom of this file. No other files need
to be modified. 

Pattern::

    class MyNewTool(BaseTool):
        ...

    # In this file, add to _TOOL_CLASSES:
    from mcp.tools.my_new_tool import MyNewTool
    _TOOL_CLASSES.append(MyNewTool)
"""

from __future__ import annotations

from mcp.schemas import ToolCategory, ToolDefinition, ToolMetadata
from mcp.tools.base import BaseTool
from utils.logger import get_logger

log = get_logger(__name__)


class ToolRegistry:
    """
    Holds all registered tools, indexed by name.

    The registry stores ToolMetadata (schema + config) separately from
    the tool instances (Python classes with execute() methods). This
    allows the registry to answer "what tools are available?" without
    importing heavy dependencies (Google API etc.) at startup.
    """

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}
        self._metadata: dict[str, ToolMetadata] = {}

    # registration
    def register(
        self,
        tool: BaseTool,
        category: ToolCategory = ToolCategory.SYSTEM,
        requires_auth: bool = False,
        timeout_seconds: float = 30.0,
        is_destructive: bool = False,
    ) -> None:
        """
        Registers a tool instance in the registry.

        Args:
            tool:             An instantiated BaseTool subclass.
            category:         Semantic grouping for filtering.
            requires_auth:    If True, check credentials before using this tool.
            timeout_seconds:  Max execution time before the dispatcher times out.
            is_destructive:   True for tools that modify/delete data.
        """
        name = tool.definition.name
        if name in self._tools:
            log.warning("tool_already_registered", name=name, action="overwriting")

        self._tools[name] = tool
        self._metadata[name] = ToolMetadata(
            definition=tool.definition,
            category=category,
            requires_auth=requires_auth,
            timeout_seconds=timeout_seconds,
            is_destructive=is_destructive,
        )
        log.info("tool_registered", name=name, category=category.value)

    # lookup func.(s)

    def get_tool(self, name: str) -> BaseTool | None:
        return self._tools.get(name)

    def get_metadata(self, name: str) -> ToolMetadata | None:
        return self._metadata.get(name)

    def has_tool(self, name: str) -> bool:
        return name in self._tools

    # the schema export (sent to llm) 

    def get_openai_tools(self, category: ToolCategory | None = None) -> list[dict]:
        """
        Return all enabled tools in OpenAI function-calling format.
        This is what gets passed to llama.cpp's 'tools' parameter.

        Args:
            category: If provided, only return tools in this category.
        """
        result = []
        for name, meta in self._metadata.items():
            if not meta.is_enabled:
                continue
            if category and meta.category != category:
                continue
            result.append(meta.definition.to_openai_format())
        return result

    def get_all_definitions(self) -> list[ToolDefinition]:
        return [m.definition for m in self._metadata.values() if m.is_enabled]

    # management:-

    def enable(self, name: str) -> None:
        if name in self._metadata:
            self._metadata[name].is_enabled = True
            log.info("tool_enabled", name=name)

    def disable(self, name: str) -> None:
        if name in self._metadata:
            self._metadata[name].is_enabled = False
            log.info("tool_disabled", name=name)

    def list_tools(self) -> list[dict]:
        return [
            {
                "name": name,
                "description": meta.definition.description[:80],
                "category": meta.category.value,
                "enabled": meta.is_enabled,
                "destructive": meta.is_destructive,
            }
            for name, meta in self._metadata.items()
        ]

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools


# singleton registry + initialisation

_registry: ToolRegistry | None = None


def get_registry() -> ToolRegistry:
    if _registry is None:
        raise RuntimeError("ToolRegistry not initialised.")
    return _registry


def init_registry() -> ToolRegistry:
    """
    Creates the registry and registers all tools.
    Add new tool classes here!
    """
    global _registry
    _registry = ToolRegistry()

    # register tools

    # import here to avoid circular imports and keep startup lazy
    try:
        from mcp.tools.google_calendar import (
            ListCalendarEventsTool,
            CreateCalendarEventTool,
            UpdateCalendarEventTool,
            DeleteCalendarEventTool,
            GetCalendarEventTool,
        )
        _registry.register(
            ListCalendarEventsTool(),
            category=ToolCategory.CALENDAR,
            requires_auth=True,
        )
        _registry.register(
            GetCalendarEventTool(),
            category=ToolCategory.CALENDAR,
            requires_auth=True,
        )
        _registry.register(
            CreateCalendarEventTool(),
            category=ToolCategory.CALENDAR,
            requires_auth=True,
            is_destructive=False,
        )
        _registry.register(
            UpdateCalendarEventTool(),
            category=ToolCategory.CALENDAR,
            requires_auth=True,
            is_destructive=True,
        )
        _registry.register(
            DeleteCalendarEventTool(),
            category=ToolCategory.CALENDAR,
            requires_auth=True,
            is_destructive=True,
            timeout_seconds=15.0,
        )
        log.info("google_calendar_tools_registered")
    except Exception as exc:
        log.warning("google_calendar_tools_failed", error=str(exc),
                    hint="Check Google API credentials and install google-api-python-client")

    log.info("tool_registry_ready", total_tools=len(_registry))
    return _registry
