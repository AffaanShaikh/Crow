"""
Tool Router - decides which tools to inject for a given user message.

Why this exists:- 
Without routing, every message was getting ALL tools injected into the LLM context,
causing:
    - tool calling for normal conversation
    - tool schemas bloat in the context, thereby causing model's persona to steer away

    Router acts as a gatekeeper: "does this message need any tools, and if so, which categories?",
    only matching tools are passed to the agentic loop.

Scalable:-
    Router never hardcodes tool names, works from ToolCategory values and
    auto-derives which categories are active from the registry. 
    
The routing strategy:- 
2-tier, zero-extra-latency for common cases
    Tier 1 - Keyword match (< 0.1 ms):
        Pattern sets per category. Instant, handles ~95% of real requests.
        "List my events"    -> ["calendar"]
        "Whatssup"          -> []

    Tier 2 - LLM classification (a quick API call):
        Used ONLY when tier 1 is ambiguous (message is long, indirect, or complex).
        Example: "I'm worried I'll miss the appointment tomorrow" -> might need calendar.
        Triggered only when the message is >13 words AND tier 1 returned nothing.
        max_tokens=20, temp=0.0 -> adds ~200-400ms only for ambiguous messages.

Adding later tools, like Spotify later we'll do:
    - in registry.py - register Spotify tools with category=ToolCategory.SPOTIFY
    - in router.py - add SPOTIFY to CATEGORY_TRIGGERS below
"""

from __future__ import annotations

import json
import re
from typing import Any
from config import get_settings
from mcp.schemas import ToolCategory
from utils.logger import get_logger

log = get_logger(__name__)
settings = get_settings()


# Per-category keyword triggers for Tier 1 routing strategy,
# patterns are matched against the lowercased user message

CATEGORY_TRIGGERS: dict[str, list[str]] = {
    ToolCategory.CALENDAR.value: [  
        "calendar", "schedule", "event", "appointment", "meeting", 
        "book a", "reschedule", "cancel my", "move my", "create an event",
        "add to my", "set a reminder", "set up a", "block time", 
        "what's on", "what do i have", "what am i doing", "agenda",
        "upcoming", "this week", "next week", "today's", "tomorrow",
        "when is my", "when am i", "am i free", "free slot",
        "list.*event", "show.*event", "my events",
        "delete.*event", "remove.*event", "update.*event", "edit.*event", # CRUD verbs next to time/event words
        "change.*time", "change.*date",
    ], 
    ToolCategory.SPOTIFY.value: [
        # Explicit controls
        r"\bplay\b", r"\bpause\b", r"\bresume\b", r"\bskip\b",
        r"\bnext\s+track\b", r"\bprevious\s+track\b", r"\bnext\s+song\b",
        r"\bprev(ious)?\s+song\b", r"\bstop\s+music\b",
        # Settings
        r"\bshuffle\b", r"\brepeat\b", r"\bloop\b",
        r"\bvolume\b",
        # Object nouns
        r"\bspotify\b", r"\bplaylist\b", r"\bsong\b", r"\btrack\b",
        r"\balbum\b", r"\bartist\b",
        # Query phrases
        r"what.{0,15}(playing|song|track)",
        r"(play|put on|queue)\s+(some\s+)?music",
        r"play\s+(a\s+)?(song|track|album|playlist)",
    ],
    # ToolCategory.SPOTIFY.value: [
    #     "play", "pause", "skip", "music", "song", "playlist", "shuffle",
    #     "volume", "spotify", "queue", "now playing",
    # ],
    # ToolCategory.COMMUNICATION.value: [
    #     "email", "send a message", "gmail", "reply to", "draft",
    # ],
}

# The ambiguity threshold, i.e. only run Tier 2 routing strategy (llm classification)
# if message exceeds this word count AND keyword tier returned nothing
AMBIGUITY_WORD_THRESHOLD = 1

# system prompt for Tier 2 routing strategy (llm classification)
# _CLASSIFIER_SYSTEM = """\
# Act as a request classifier. Given a user message, determine which tool \
# categories are needed to fulfill it.

# **Try to find if a given tool can fulfill the user's query and return that.**

# Available categories and what they cover:
# {categories}

# IMPORTANT Rules:
# - Return ONLY a JSON array of matching category names, e.g. ["calendar"]
# - Return [] for greetings, opinions, general questions, personal conversation
# - Return [] if the request can be answered from general knowledge alone
# - Only include a category if the user is EXPLICITLY asking to read or modify that data

# Examples for how you should answer:
# "hello" -> []
# "who are you?" -> []
# "what should I eat today?" -> []
# "list my calendar events" -> ["calendar"]
# "am I free tomorrow afternoon?" -> ["calendar"]
# "schedule a dentist appointment for Friday" -> ["calendar"] 
# """
_CLASSIFIER_SYSTEM = """\
Act as a tool selector and executor. Given: User's recent message(s) and a 'tools' list. **Determine which tools can fulfill user's query/request. \
Search thoroughly throughout the entire message(s) of the user to determine if he/she is trying to use one of the tools. \

- Use ONLY tools from the tool list. \
- Return ONLY a JSON array of matching category name only if tool name is in the list. \
- Reason efficiently and quickly to answer.** \

Tools list:
{categories}

Examples for how you should answer:
"hello" -> []
"who are you?" -> []
"let's discuss movies" -> []
"play some breakcore from my saved Breakcore playlist" -> ["spotify"]
"will I be able to make it there tommorrow at 7 pm?" -> ["calendar"]
"I wonder if I should take an umbrella with me" -> ["weather"] 
"""


class ToolRouter:
    """
    Routes user messages to relevant tool categories.

    Usage::

        router = ToolRouter(llm_client, registry)
        tools = await router.get_tools_for_message("list my events")
        # -> [{"type": "function", "function": {...list_calendar_events...}}]

        tools = await router.get_tools_for_message("hello!")
        # -> []
    """

    def __init__(self, llm_client: Any, registry: Any) -> None:
        self.llm = llm_client
        self.registry = registry
        # build the human-readable category descriptions from the registry
        self._category_descriptions = self._build_category_descriptions()
        log.info(
            "tool_router_init",
            active_categories=list(self._category_descriptions.keys()),
        )

    # public api

    async def get_tools_for_messages(self, messages: list[list[str]]) -> list[dict]:
        """
        Return OpenAI-format tool schemas relevant to this message.
        Returns [] for non-tool requests - agent loop becomes a plain chat call.
        """
        categories = await self._classify(messages)
        if not categories:
            log.debug("router_no_tools", message_preview=messages[0][:60])
            return []

        tools: list[dict] = []
        for cat_name in categories:
            try:
                cat = ToolCategory(cat_name)
                tools.extend(self.registry.get_openai_tools(category=cat))
            except ValueError:
                log.warning("router_unknown_category", name=cat_name)

        log.info(
            "router_decision",
            message_preview=messages[0][:60],
            matched_categories=categories,
            tools_injected=len(tools),
        )
        return tools

    # classification

    async def _classify(self, messages: list[list[str]]) -> list[str]:
        """
        Two-tier classification.
            Tier 1: keyword match (~instant).
            Tier 2: LLM call (only for long ambiguous messages where tier 1 found nothing).
        """
        # Tier 1 - keyword
        matched = self._keyword_match(messages[0]+messages[1])
        if matched:
            log.debug("router_tier1_keyword_match", categories=matched)
            return matched

        # Tier 2 - LLM
        word_count = len(messages[0].split() + messages[1].split())
        if word_count > AMBIGUITY_WORD_THRESHOLD and self._category_descriptions:
            llm_result = await self._llm_classify(messages)
            if llm_result:
                log.debug("router_tier2_llm_match", categories=llm_result)
            return llm_result
        
        # short message, no keywords -> treat as conversation
        return []

    def _keyword_match(self, message: str) -> list[str]:
        """
        Checks message against per-category pattern lists.
        Uses regex so patterns like "list.*event" work correctly.
        """
        msg_lower = message.lower()
        matched: list[str] = []

        for cat_name, patterns in CATEGORY_TRIGGERS.items():
            # check only those categories that have registered tools
            if cat_name not in self._category_descriptions:
                continue
            for pattern in patterns:
                if re.search(pattern, msg_lower):
                    matched.append(cat_name)
                    break  # one match per category is enough
        return matched

    async def _llm_classify(self, messages: list[list[str]]) -> list[str]:
        """
        LLM-based classification for determining if any and which tools \
        should be called based on user's last {limit} messages.
        Returns list of category names, or [] on any failure.
        """
        if not self._category_descriptions:
            return []

        cat_lines = "\n".join(
            f'  "{name}": {desc}'
            for name, desc in self._category_descriptions.items()
        )
        system = _CLASSIFIER_SYSTEM.format(categories=cat_lines)

        try:
            recent_messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": messages[0]},
                    {"role": "system", "content": system},
                    {"role": "user", "content": messages[1]},
                ]
            params = {                                      # thinking params:-
                "model":       settings.llm_model_name,
                "messages":    recent_messages,
                "max_tokens":  512,
                "temperature": 0.6,
                "top_p":       settings.default_top_p,
            }
            extra_body = {}
            extra_body["top_k"] = 20
            extra_body["min_p"] = 0
            extra_body["reasoning_effort"] = "low"
            if extra_body:
                params["extra_body"] = extra_body
    
            log.info("params to _llm_classify", params=params)
            response = await self.llm._client.chat.completions.create(**params)
            #     model=settings.llm_model_name,
            #     messages=[
            #         {"role": "system", "content": system},
            #         {"role": "user", "content": messages[0]},
            #         {"role": "system", "content": system},
            #         {"role": "user", "content": messages[1]},
            #     ],
            #     max_tokens=50,
            #     temperature=0.1,
            # )

            raw = (response.choices[0].message.content or "").strip()
            # strip any markdown code fences the model might add
            # raw = re.sub(r"```[a-z]*\n?|\n?```", "", raw).strip()
            # result = json.loads(raw)
            # return [str(c) for c in result] if isinstance(result, list) else []
            log.info("_llm_classify response", response)
            return _parse_classifier_response(raw)
        except Exception as exc:
            log.warning("router_llm_classify_failed", error=str(exc))
            return []  # safe default - no tools -> direct persona answer
 

    def _build_category_descriptions(self) -> dict[str, str]:
        """
        Derive category descriptions from registered tools.
        Only categories with at least one enabled tool are included.
        This means the router automatically reflects the registry state.
        """
        cats: dict[str, list[str]] = {}
        for name, meta in self.registry._metadata.items():
            if not meta.is_enabled:
                continue
            cat = meta.category.value
            # if cat not in cats:
            #     cats[cat] = []
            # cats[cat].append(name)
            cats.setdefault(cat, []).append(name)
        return {cat: f"tools: {', '.join(tools)}" for cat, tools in cats.items()}

def _parse_classifier_response(raw: str) -> list[str]:
    """
    Parse llm classifier output robustly.

    Handles: valid JSON array, markdown fences, plain text, empty string.
    """
    if not raw:
        return []

    # strip markdown code fences if present
    cleaned = re.sub(r"'```[a-z]*\s*|\s*```'", "", raw).strip()

    # direct parse
    try:
        result = json.loads(cleaned)
        if isinstance(result, list):
            return [str(c) for c in result if c]
        return []
    except json.JSONDecodeError:
        pass

    # try to extract an array from anywhere in the response
    m = re.search(r"\[([^\]]*)\]", cleaned)
    if m:
        try:
            result = json.loads(f"[{m.group(1)}]")
            if isinstance(result, list):
                return [str(c) for c in result if c]
        except json.JSONDecodeError:
            pass

    # in case model returned a plain category name (rare but happens)
    for cat in ToolCategory:
        if cat.value in cleaned.lower():
            return [cat.value]

    log.debug("router_classify_unparseable", raw=raw)#[:100])
    return []


# singleton

_router: ToolRouter | None = None


def get_router() -> ToolRouter:
    if _router is None:
        raise RuntimeError("ToolRouter not initialised.")
    return _router


def init_router(llm_client: Any, registry: Any) -> ToolRouter:
    global _router
    _router = ToolRouter(llm_client, registry)
    return _router