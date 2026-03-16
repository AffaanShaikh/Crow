"""
an in-memory context manager, implements a layered memory strategy:
  - sliding window of recent raw turns
  - token budget enforcement
  - summarisation trigger (todo)
  - per-session state stored in a simple dict (todo: swap for Redis in multi-process deployment)

    thread-safety: uses asyncio.Lock per session, safe for async FastAPI workers,
    each session has its own asyncio.Lock, prevents a race condition where two simultaneous requests for same session could corrupt the turn history
"""

import asyncio
from dataclasses import dataclass, field
from typing import Optional

from config import get_settings
from utils.logger import get_logger

log = get_logger(__name__)
settings = get_settings()
 

@dataclass # autogenerates the boilerplate __init__, __repr__, etc.
class SessionState:
    """
    holds everything about one conversation session, including:
    - session_id: unique identifier for the session
    - turns: list of dicts with 'user' and 'assistant' keys representing the conversation history
    - summary: optional string that holds a compressed summary of earlier conversation turns
    - total_tokens_estimate: an integer that estimates the total tokens used in the session (for budgeting purposes)
    - lock: an asyncio.Lock to ensure thread-safe access to the session state when multiple requests are processed concurrently for the same session
    """
    session_id: str
    turns: list[dict] = field(default_factory=list)   # {"user": str, "assistant": str}
    summary: Optional[str] = None
    total_tokens_estimate: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def turn_count(self) -> int:
        return len(self.turns)

    def token_estimate(self) -> int:
        """
        a rough token budget: sum of all 'turn word-counts x tokens_per_word',
        dont wanna involve a tokenizer just yet
        """
        total_words = sum(
            len(t["user"].split()) + len(t["assistant"].split())
            for t in self.turns
        )
        if self.summary:
            total_words += len(self.summary.split())
        return int(total_words * settings.tokens_per_word)



class ContextManager:
    """
    manages per-session conversation context,
    key responsibilities:
    - store turns after each exchange
    - return only the turns that fit within the token budget
    - trigger summarisation when history grows too long
    """
    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}
        log.info("context_manager_init")

    # the actual session management methods: create/get/delete sessions, and get session info
    def get_or_create_session(self, session_id: str) -> SessionState:
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionState(session_id=session_id)
            log.info("session_created", session_id=session_id)
        return self._sessions[session_id]

    def get_session_info(self, session_id: str) -> dict:
        session = self._sessions.get(session_id)
        if not session:
            return {"exists": False}
        return {
            "exists": True,
            "turn_count": session.turn_count(),
            "token_estimate": session.token_estimate(),
            "has_summary": session.summary is not None,
        }

    def delete_session(self, session_id: str) -> None:
        if session_id in self._sessions:
            del self._sessions[session_id]
            log.info("session_deleted", session_id=session_id)


    # ctx retrieval for prompt assembly, with token budget enforcement
    async def get_context(self, session_id: str) -> tuple[list[dict], Optional[str]]:
        """
        for prompt assembly, returns (recent_turns, summary_or_None),
        trims turns to stay within the token budget.
        """
        session = self.get_or_create_session(session_id)
        async with session.lock:
            turns = self._trim_to_budget(session.turns)
            return turns, session.summary

    def _trim_to_budget(self, turns: list[dict]) -> list[dict]:
        """
        we trim by keeping the most recent turns that fit within the token budget
        & always keeps at least 1 turn so the model has some grounding.
        """
        # reserved tokens for: system prompt + few-shot + new user message 
        reserved = 800
        budget = settings.context_window_tokens - reserved - (
            int(len((self._sessions or {}).values()) * 0)  # placeholder for summary tokens
        )

        kept: list[dict] = []
        token_count = 0
        for turn in reversed(turns): # most recent turns first
            words = len(turn["user"].split()) + len(turn["assistant"].split())
            cost = int(words * settings.tokens_per_word)
            if token_count + cost > budget and kept:
                break
            kept.insert(0, turn)
            token_count += cost

        if len(kept) < len(turns):
            log.debug(
                "context_trimmed",
                original=len(turns),
                kept=len(kept),
                token_estimate=token_count,
            )
        return kept

    # turn recording and summarisation trigger (todo)
    async def add_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_response: str,
    ) -> None:
        """record a completed exchange and trigger summarisation if needed"""
        session = self.get_or_create_session(session_id)
        async with session.lock:
            session.turns.append({
                "user": user_message,
                "assistant": assistant_response,
            })
            log.debug(
                "turn_added",
                session_id=session_id,
                turn=session.turn_count(),
                token_estimate=session.token_estimate(),
            )
            if session.turn_count() >= settings.summary_trigger_turns:
                await self._kurzify(session)

    async def _kurzify(self, session: SessionState) -> None:
        """
        todo: get a llm to generate an actual semantic summary, but for now:
        keep the last 'max_history_turns' turns and store a summary string of the overflowed earlier turns
        """
        if session.turn_count() <= settings.max_history_turns:
            return
 
        overflow = session.turns[: -settings.max_history_turns]
        session.turns = session.turns[-settings.max_history_turns :] 
        snippets = "; ".join(
            f'User asked about "{t["user"][:60].strip()}"' for t in overflow
        )
        session.summary = (
            f"[Earlier conversation covered: {snippets}. ]" 
        )
        log.info(
            "context_summarised",
            session_id=session.session_id,
            overflowed=len(overflow),
            remaining=session.turn_count(),
        )


    def active_session_count(self) -> int:
        return len(self._sessions)

    def clear_all_sessions(self) -> None: 
        self._sessions.clear()
        log.warning("all_sessions_cleared")


# singletons to manage the ctx instance, later swap for Redis (coz scalability, stateless etc.)
_context_manager: ContextManager | None = None

def get_context_manager() -> ContextManager:
    if _context_manager is None:
        raise RuntimeError("ContextManager not initialised.")
    return _context_manager

def init_context_manager() -> ContextManager:
    global _context_manager
    _context_manager = ContextManager()
    return _context_manager
