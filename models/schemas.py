"""
Pydantic v2 schemas for API request/response models
"""
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


# enums
class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

class StreamEventType(str, Enum):
    DELTA = "delta"         # le partial token
    DONE = "done"           # stream finished
    ERROR = "error"         # something went wrong
    METADATA = "metadata"   # session info etc.

class HealthStatus(str, Enum):
    OK = "ok"
    DEGRADED = "degraded"
    DOWN = "down"


# core message types
class Message(BaseModel):
    role: Role
    content: str

    model_config = {"use_enum_values": True}


class ConversationTurn(BaseModel):
    """a single user <-> assistant exchange stored in memory"""
    user: str
    assistant: str
    token_estimate: int = 0


# chat api
class ChatRequest(BaseModel):
    session_id: str = Field(
        ...,
        description="Client-generated UUID for this conversation session.",
        examples=["550e8400-e29b-41d4-a716-446655440000"],
    )
    message: str = Field(
        ...,
        min_length=1,
        max_length=8000,
        description="The user's latest message.",
    )
    # per-request overrides (optional)
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, ge=1, le=4096)
    stream: Optional[bool] = None


class ChatResponse(BaseModel):
    """returned for non-streaming responses"""
    session_id: str
    message: str
    model: str
    usage: dict[str, int]


class StreamEvent(BaseModel):
    """
    the payload for each (SSE) Server-Sent Event chunk,
    frontend will parse 'data:' lines and reconstructs these
    """
    type: StreamEventType
    content: Optional[str] = None # the delta text
    session_id: Optional[str] = None
    error: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None

    model_config = {"use_enum_values": True}


# Gesündheit! 
class ComponentHealth(BaseModel):
    status: HealthStatus
    detail: Optional[str] = None
    latency_ms: Optional[float] = None

    model_config = {"use_enum_values": True}

class HealthResponse(BaseModel):
    status: HealthStatus
    version: str
    components: dict[str, ComponentHealth]
    feature_flags: dict[str, bool]

    model_config = {"use_enum_values": True}


# session schema - what we'll store in memory for each conversation
class SessionInfo(BaseModel):
    session_id: str
    turn_count: int
    token_estimate: int
    has_summary: bool
    feature_flags: dict[str, bool]
