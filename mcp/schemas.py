"""
MCP (Model Context Protocol) data schemas.

These types define the complete contract for tool-calling in this system,
compatible with both:
  - The OpenAI function-calling API format (used to send tools to llama.cpp)
  - The Anthropic MCP specification (for future MCP server/client mode)

Data flow:
  1. ToolDefinition   -> sent to LLM as "available tools"
  2. ToolCall         -> LLM decides to call a tool (parsed from LLM response)
  3. ToolResult       -> what the tool returned (injected back into context)
  4. AgentStep        -> one iteration of the agent loop (call + result pair)
  5. AgentResponse    -> final assembled response after all tool calls complete 

"""

from __future__ import annotations

from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


# tool Definition, what we tell the llm

class ParameterProperty(BaseModel):
    """Describes one parameter of a tool in JSON Schema format."""
    type: str                           # "string", "integer", "boolean", "array", "object"
    description: str
    enum: list[str] | None = None       # allowed values
    default: Any = None
    items: dict | None = None           # for array types, describes item type
    format: str | None = None           # e.g. "date-time", "email"
    minimum: float | None = None
    maximum: float | None = None


class ToolParameters(BaseModel):
    """JSON Schema object describing a tool's parameters."""
    type: str = "object"
    properties: dict[str, ParameterProperty]
    required: list[str] = Field(default_factory=list)
    additionalProperties: bool = False


class ToolDefinition(BaseModel):
    """
    A tool the llm can choose to call,
    serialised to OpenAI-compatible format when sent to llama.cpp.
    """
    name: str = Field(
        ...,
        pattern=r"^[a-zA-Z0-9_-]{1,64}$",
        description="Snake_case tool name. Must be unique in the registry.",
    )
    description: str = Field(
        ...,
        description=(
            "Clear, specific description of what this tool does, when to use it, "
            "and what it returns. The llm uses this to decide whether to call it."
        ),
    )
    parameters: ToolParameters

    def to_openai_format(self) -> dict:
        """Converts to OpenAI function-calling tool format for llama.cpp API."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters.model_dump(exclude_none=True),
            },
        }


# tool call, what the LLM asks for

class ToolCallStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"


class ToolCall(BaseModel):
    """
    A tool invocation requested by the LLM.
    Parsed from the LLM's response message.
    """
    id: str                     # OpenAI-style call ID (ex: "call_jkl789")
    tool_name: str
    arguments: dict[str, Any]   # parsed from JSON string
    status: ToolCallStatus = ToolCallStatus.PENDING


class ToolResult(BaseModel):
    """The output of executing a ToolCall."""
    call_id: str
    tool_name: str
    success: bool
    output: Any                 # any JSON-serialisable value
    error: str | None = None
    execution_ms: float = 0.0

    def to_message_content(self) -> str:
        """
        Format the result for injection into the LLM context.
        The LLM needs to read and interpret this.
        """
        if self.success:
            import json
            return json.dumps(self.output, indent=2, default=str)
        return f"ERROR: {self.error}"


# agent loop types

class AgentStepType(str, Enum):
    THINKING = "thinking"       # LLM reasoning before tool call
    TOOL_CALL = "tool_call"     # LLM decided to call a tool
    TOOL_RESULT = "tool_result" # tool execution result
    FINAL = "final"             # LLM produced final answer


class AgentStep(BaseModel):
    """One step in the agent's reasoning chain."""
    type: AgentStepType
    content: str | None = None              # text content (for THINKING/FINAL)
    tool_call: ToolCall | None = None       # populated for TOOL_CALL
    tool_result: ToolResult | None = None   # populated for TOOL_RESULT


class AgentResponse(BaseModel):
    """
    The complete result of one agent invocation.
    Contains the final answer and the full reasoning chain for transparency.
    """
    final_answer: str
    steps: list[AgentStep]
    total_tool_calls: int
    session_id: str

    @property
    def used_tools(self) -> list[str]:
        return [s.tool_call.tool_name for s in self.steps if s.tool_call]


# tool metadata for registry

class ToolCategory(str, Enum):
    CALENDAR = "calendar"
    COMMUNICATION = "communication"
    KNOWLEDGE = "knowledge"
    SYSTEM = "system"
    WEB = "web"


class ToolMetadata(BaseModel):
    """Extended metadata stored in the registry alongside the definition."""
    definition: ToolDefinition
    category: ToolCategory
    requires_auth: bool = False
    timeout_seconds: float = 30.0
    is_destructive: bool = False    # True for delete/modify operations
    is_enabled: bool = True
