"""
Agent Loop - 2-phase agentic reasoning.

Reasons for the 2 Phases arch.:-
    Using the persona context + few-shot examples during tool calls confuses
    the model - it sees emotional conversation examples alongside calendar tools
    and hallucinates. With a stripped, purpose-built system prompt for tool calls:-
    - Tool selection is at temp=0.0 -> deterministic and accurate
    - No persona bleed-through during tool logic
    - Final answer gets the full persona at proper temperature -> sounds like Elda

The loop can run multiple cycles ("turns" internally) when the llm calls tools in sequence.

ARCHITECTURE
────────────
                    user message
                         │
                    ToolRouter
              (keyword + LLM classify)
                         │
           ┌─────────────┴─────────────┐
       no tools                   tools matched
           │                           │
    Stream persona direct       Phase 1: Tool Orchestration
    (DELTA per token)           ─────────────────────────────
           │                   Stripped context, temp=0.0
           │                   Loop until no more tool_calls
           │                           │
           │                  Phase 2: Streaming Synthesis
           │                  ─────────────────────────────
           │                  Full persona context
           │                  Inject: live tool results
           │                  Persona temperature
           │                  (DELTA per token)
           │                           │
           └─────────────┬─────────────┘
                    FINAL AgentStep
              (content = complete assembled text,
               used for memory - NOT re-emitted)

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

import re
import json 
from collections.abc import AsyncIterator
from typing import Any

from config import get_settings
from mcp.dispatcher import ToolDispatcher, parse_tool_calls_from_response
from mcp.registry import ToolRegistry
from mcp.router import ToolRouter#, get_tools_for_messages
from mcp.schemas import AgentResponse, AgentStep, AgentStepType
from models.schemas import Message, Role
from utils.logger import get_logger

log = get_logger(__name__)
settings = get_settings()

MAX_ITERATIONS = 8              # hard cap on tool-call cycles per user message
MAX_TOOL_CALLS_PER_STEP = 5     # max parallel tool calls in one iteration

# # Qwen3 think-tag patterns
# _THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

# Phase 1 system prompt: tool orchestration only
# Built dynamically so the current UTC time is always injected
# Shown ONLY to tool-calling calls, never to the final persona synthesis

def _build_orchestration_system() -> str:
    """Builds the tool-calling system prompt with live UTC timestamp for tool-calling calls."""
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    return (
        f"You are a precise tool-calling agent. Call tools correctly to fulfill the request.\n\n"
        f"CURRENT UTC TIME: {now.strftime('%A %Y-%m-%d %H:%M')} UTC\n\n"
        "GENERAL RULES:\n"
        "- DO NOT write conversational text - ONLY call tools\n"
        "- Required parameters must come from the user message or prior tool results\n"
        "- Omit optional parameters the user did not provide (never pass empty strings)\n"
        "- Stop calling tools once you have all necessary data\n"
        "RULES FOR TOOLS:-"
        "DATETIME RULES (critical):\n"
        "- All datetime fields MUST be ISO 8601: YYYY-MM-DDTHH:MM:SS+00:00\n"
        "- Use the current UTC time above to compute dates from user language:\n"
        "    \'tomorrow at 7pm\' -> compute the exact ISO datetime\n"
        "    \'this Friday\' -> compute Friday of this week as ISO datetime\n"
        "    \'next week\' -> compute Monday of next week as ISO datetime\n"
        "- NEVER pass natural language into datetime fields\n\n"
        # For time/calendar-based queries:-
        # - Accurately set 'time_min' and 'time_max' in ISO-8601 format based on user's prompt.
        # - Based on today's date correctly interpret the 'time_min' and 'time_max' being asked by the user.
        # - For "this week" / "today" / "upcoming" queries, DO NOT specify 'time_min' or 'time_max' - let the tool use its defaults
        # - Only specify time ranges when the user gives specific dates (ex., "events in July", "meetings on March 20th"), pay attention to the year too ofcourse.
        "ATTENDEES RULES:\n"
        "- attendees must be a JSON array of email addresses containing @ (e.g. [\"alice@x.com\"])\n"
        "- Omit attendees entirely if no email address was explicitly given\n\n"
        "EVENT_ID RULES (critical):\n"
        "- event_id is a ~26-char alphanumeric string from the Google Calendar API\n"
        "- NEVER invent, guess, or use an event title as event_id\n"
        "- For update or delete: call list_calendar_events FIRST to get the real event_id,\n"
        "  then call update/delete with that id\n\n"
    )


# Phase 2 synthesis injection: appended to persona context,
# injected as a system message immediately before the user's message so the
# persona LLM sees: [persona system] -> [history] -> [tool context] -> [user msg]

# SYNTHESIS_CONTEXT_TEMPLATE = """\
# Today is {today_utc} (UTC).

# The following data was retrieved live to answer the user's request.
# Report ONLY what appears in the data below - do not add, invent, or recall any \
# events, items, or details that are not explicitly listed here.
# If the results are empty, say so honestly.
# Answer in your own voice and persona. Do not mention that you used a tool.

# {tool_summary}
# """
SYNTHESIS_CONTEXT_TEMPLATE = """\
TOOLS RETURNED RESULTS:- \
Today is {today_utc} (UTC). \
{tool_summary}
"""


class AgentLoop:
    """
    Orchestrates multi-step tool-use reasoning. (over an OpenAI-compatible llm)
    w/ two-phase separation:
        tool calls (stripped/deterministic) vs final answer (persona/expressive).
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
        router: ToolRouter,
    ) -> None:
        self.llm = llm_client
        self.dispatcher = dispatcher
        self.registry = registry
        self.router = router
        self.max_iterations = settings.mcp_max_iterations

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
        # final_text: list[str] = []
        final_text = ""

        async for step in self.run_streaming(messages, session_id):
            steps.append(step)
            if step.type == AgentStepType.FINAL and step.content:
                # final_text.append(step.content)
                final_text = step.content  # authoritative assembled text

        return AgentResponse(
            # final_answer="".join(final_text) or "Unable to complete this request.",
            final_answer=final_text or "Unable to complete this request.",
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
        Streaming agent loop. Yields AgentStep events as the agent reasons.

        The frontend subscribes to these to show live reasoning:
          - THINKING        -> LLM is producing text before a tool call
          - TOOL_CALL       -> LLM requested a tool, one per tool call requested
          - TOOL_RESULT     -> tool executed, result available, one per executed tool
          - FINAL           -> LLM's final answer, streams this token-by-token

        Converts Message objects -> dicts, injects tool-use addendum into
        the system message, then loops until no more tool calls.
        """
        # user's latest message extracted for routing
        last_user_msg = _last_user_message(messages)
        last_user_msg_two = _last_user_messages(messages, limit=2)
        # log.info("users_last_two_messages", last_user_msg_two)
        # Route: which tools does this message need?
        relevant_tools = await self.router.get_tools_for_messages(last_user_msg_two) #(last_user_msg)

        log.info(
            "agent_loop_start",
            session_id=session_id,
            tools_injected=len(relevant_tools),
            message_count=len(messages),
            model=settings.llm_model_name,
        )

        # # path A: no tools needed -> direct persona response (streaming)
        # if not relevant_tools:
        #     log.debug("agent_direct_path_wStreaming", session_id=session_id)

        #     full_response: list[str] = []
        #     # log.info("messages in agent_direct_path_wStreaming", messages)
        #     async for token in self._stream_persona_direct(messages):
        #         full_response.append(token)
        #         yield AgentStep(type=AgentStepType.DELTA, content=token)#.THINKING, content=token)

        #     # log.info("full_response in agent_direct_path_wStreaming", full_response)
        #     yield AgentStep(
        #         type=AgentStepType.FINAL,
        #         content="".join(full_response),
        #     )
        #     return
        # path A: no tools -> RAG retrieval + stream through persona
        if not relevant_tools:
            log.debug("agent_direct_path_streaming", session_id=session_id)

            # RAG retrieval: inject context even on the direct path
            rag_ctx: str | None = None
            if settings.rag_enabled:
                try:
                    from rag.retriever import get_retriever
                    rag_result = await get_retriever().retrieve(
                        query=last_user_msg,
                        collection=settings.rag_default_collection,
                    )
                    if rag_result.context_str:
                        rag_ctx = rag_result.context_str
                        log.info(
                            "rag_context_injected_path_a",
                            session_id=session_id,
                            chunks=len(rag_result.results),
                        )
                except Exception as exc:
                    log.warning("rag_retrieval_failed_path_a", error=str(exc))

            # if RAG returned context, we rebuild messages with it injected
            if rag_ctx:
                from llm.prompt_builder import build_messages as _bm_a
                from memory.context_manager import get_context_manager
                last_msg = _last_user_message(messages)
                history_a: list[dict] = []
                msgs_a = list(messages)
                i = 0
                while i < len(msgs_a) - 1:
                    if msgs_a[i].role == "user" and i + 1 < len(msgs_a) and msgs_a[i + 1].role == "assistant":
                        history_a.append({"user": msgs_a[i].content, "assistant": msgs_a[i + 1].content})
                        i += 2
                    else:
                        i += 1
                log.info("history&summary run_streaming", history=history_a, rag_context=rag_ctx)
                rag_messages = _bm_a(
                    user_message=last_msg,
                    history_turns=history_a,
                    rag_context=rag_ctx,
                )
                use_messages = rag_messages
            else:
                use_messages = messages

            full: list[str] = []
            async for token in self._stream_persona_direct(use_messages):
                full.append(token)
                yield AgentStep(type=AgentStepType.DELTA, content=token)
            yield AgentStep(type=AgentStepType.FINAL, content="".join(full))
            return

        # path B: tools needed -> Phase 1 (orchestration)
        # use a stripped working context - no persona, no history, just the task
        # system prompt rebuild each call so UTC timestamp is always fresh, i.e. llm knows whats up
        working: list[dict] = [
            {"role": "system", "content": _build_orchestration_system()},
            {"role": "user", "content": last_user_msg},
        ]
        accumulated_results: list = []

        for iteration in range(self.max_iterations):
            log.debug("agent_iteration", iteration=iteration, session_id=session_id)

            response_msg = await self._call_for_tools(working, relevant_tools)
            text_content: str = response_msg.get("content") or ""
            tool_calls_raw: list = response_msg.get("tool_calls") or []

            # if model ignored tools (likely unsupported model): warning
            if iteration == 0 and not tool_calls_raw and len(text_content) > 50:#150:
                log.warning(
                    "model_ignored_tools",
                    model=settings.llm_model_name,
                    hint=(
                        "Model returned text despite tools being available. "
                        "Ensure you are using a tool-calling model like: "
                        "llama3.1, llama3.2, qwen2.5, mistral-nemo or firefunction-v2"
                    ),
                )

            # no more tool calls -> Phase 1 complete
            if not tool_calls_raw:
                log.info(
                    "agent_tools_done",
                    iteration=iteration,
                    session_id=session_id,
                    results_collected=len(accumulated_results),
                )
                break

            # # emit thinking text, if present
            # if text_content.strip():
            #     yield AgentStep(type=AgentStepType.THINKING, content=text_content)
            # emit thinking text, if present (strip think tags first)
            clean_thinking = (text_content).strip() # _strip_think_tags(text_content).strip()
            if clean_thinking:
                yield AgentStep(type=AgentStepType.THINKING, content=clean_thinking)

            # parse and emit tool call events
            calls = parse_tool_calls_from_response(response_msg)
            calls = calls[:MAX_TOOL_CALLS_PER_STEP]
            for call in calls:
                yield AgentStep(type=AgentStepType.TOOL_CALL, tool_call=call)

            # record assistant message (w/ tool_calls) in working history
            working.append(response_msg)

            # execute tools in parallel
            results = await self.dispatcher.dispatch_many(calls)
            accumulated_results.extend(results)

            # inject results back as tool-role messages
            for result in results:
                yield AgentStep(type=AgentStepType.TOOL_RESULT, tool_result=result)
                working.append({
                    "role": "tool",
                    "tool_call_id": result.call_id,
                    "content": result.to_message_content(),
                })
        else:
            log.warning("agent_max_iterations", session_id=session_id)

        # # Phase 2: final synthesis with persona, (non-streaming / .complete())
        # # injects tool results into the full persona context, then calls the
        # # model at persona's temperature to sound like her
        # final_answer = await self._synthesise_with_persona(
        #     messages, accumulated_results
        # )
        # log.info(
        #     "agent_loop_done",
        #     session_id=session_id,
        #     answer_chars=len(final_answer),
        #     tools_used=len(accumulated_results),
        # )
        # yield AgentStep(type=AgentStepType.FINAL, content=final_answer)


        # # Phase 2: final synthesis with persona, (streaming)
        # full_response: list[str] = []

        # async for token in self._stream_synthesise_with_persona(
        #     messages, accumulated_results
        # ):
        #     full_response.append(token)

        #     yield AgentStep(
        #         type=AgentStepType.DELTA,#.THINKING,
        #         content=token,
        #     )

        # log.info(
        #     "agent_loop_done_wStreaming",
        #     session_id=session_id,
        #     answer_chars=len(full_response),
        #     tools_used=len(accumulated_results),
        # )

        # yield AgentStep(
        #     type=AgentStepType.FINAL,
        #     content="".join(full_response),
        # )
        # Phase 2: RAG retrieval + streaming synthesis
        rag_context: str | None = None
        if settings.rag_enabled:
            try:
                from rag.retriever import get_retriever
                last_msg = _last_user_message(messages)
                rag_result = await get_retriever().retrieve(
                    query=last_msg,
                    collection=settings.rag_default_collection,
                )
                if rag_result.context_str:
                    rag_context = rag_result.context_str
                    log.info(
                        "rag_context_injected",
                        session_id=session_id,
                        chunks=len(rag_result.results),
                        retrieval_ms=rag_result.retrieval_ms,
                    )
            except Exception as exc:
                log.warning("rag_retrieval_failed", error=str(exc))

        full: list[str] = []
        async for token in self._stream_synthesise_with_persona(
            messages, accumulated_results, rag_context=rag_context
        ):
            full.append(token)
            yield AgentStep(type=AgentStepType.DELTA, content=token)

        log.info(
            "agent_loop_done_wStreaming",
            session_id=session_id,
            answer_chars=len(full),
            tools_used=len(accumulated_results),
        )
        yield AgentStep(type=AgentStepType.FINAL, content="".join(full))


    # llm call helpers

    # async def _call_persona_direct(self, messages: list[Message]) -> str:
    #     """
    #     No-tools path - full persona, streaming-compatible complete() call.
    #     Uses persona temperature for natural, in-character responses.
    #     """
    #     text, _ = await self.llm.complete(
    #         messages,
    #         temperature=settings.default_temperature,
    #         max_tokens=settings.default_max_tokens,
    #     )
    #     return text

    async def _stream_persona_direct(
        self,
        messages: list[Message],
    ) -> AsyncIterator[str]:
        """
        Path A: stream directly through the full persona context.
        Filters qwen3 <think> blocks from the token stream.
        """
        log.info("stream direct persona", final_prompt_to_llm=messages)
        raw_stream = self.llm.stream(       # non-thinking params:-
            messages,
            max_tokens=settings.default_max_tokens,
            temperature=0.7, #settings.default_temperature,
            top_p=0.8,
            top_k=20,
            min_p=0,
            reasoning_effort="none"
        )
        async for token in raw_stream: # _strip_thinking(raw_stream):
            yield token

    # async def _synthesise_with_persona(
    #     self,
    #     original_messages: list[Message],
    #     tool_results: list,
    # ) -> str:
    #     """
    #     Phase 2 - inject tool results into the persona context, then call
    #     at persona temperature so the final answer sounds like the character.

    #     Message layout sent to llm looks like:-
    #       [system (persona)]
    #       [few-shot examples]
    #       [conversation history]
    #       [system (tool context injection)]     <- inserted here
    #       [user (current message)]
    #     """
    #     if not tool_results:
    #         # tools were routed but all failed or nothing was collected then,
    #         # fall back to a direct response
    #         return await self._call_persona_direct(original_messages)

    #     from datetime import datetime, timezone as _tz
    #     today_utc = datetime.now(_tz.utc).strftime("%A, %B %d %Y at %H:%M UTC")
    #     tool_summary = _format_tool_results(tool_results)
    #     injection_content = SYNTHESIS_CONTEXT_TEMPLATE.format(
    #         today_utc=today_utc,
    #         tool_summary=tool_summary
    #     )

    #     # the tool context injection inserted before the last user message
    #     # so the model sees it as fresh context, not a system override
    #     augmented: list[Message] = []
    #     for i, msg in enumerate(original_messages):
    #         # before the last user message, inject tool context
    #         is_last = (i == len(original_messages) - 1)
    #         if is_last and msg.role == "user":
    #             augmented.append(
    #                 Message(role=Role.SYSTEM, content=injection_content)
    #             )
    #         augmented.append(msg)

    #     # lower temp for synthesis: keeps persona voice, prevent factual hallucination (coz higher val. is causing hallucinated ans.(s))
    #     synthesis_temp = min(settings.default_temperature, 0.4)
    #     text, _ = await self.llm.complete(
    #         augmented,
    #         temperature=synthesis_temp,
    #         max_tokens=settings.default_max_tokens,
    #     )
    #     return text

    async def _stream_synthesise_with_persona(
        self,
        original_messages: list[Message],
        tool_results: list,
        rag_context: str | None = None,
    ) -> AsyncIterator[str]:
        """
        Phase 2 - inject tool results (and RAG context) into the full persona context, then stream
        at reduced temperature to prevent factual hallucination while keeping voice.

        Message layout:
          [system: safety rules]
          [system: persona]
          [few-shot examples]
          [conversation history]
          [system: rag_context]             <- RAG retrieved documents (if any)
          [system: tool context injection]  <- tool results (if any)
          [user: current message]
        """
        if not tool_results and not rag_context:
            # No enrichment - stream directly through persona
            async for token in self._stream_persona_direct(original_messages):
                yield token
            return

        from datetime import datetime, timezone as _tz
        today_utc = datetime.now(_tz.utc).strftime("%A, %B %d %Y at %H:%M UTC")
        r: str | None = ""  # !
        raw_tool_results: str | None = f"raw tool(s) output: {r}"
        raw_tool_results.format(r=tool_results)
        tool_summary = _format_tool_results(tool_results) + raw_tool_results

        injection_content = SYNTHESIS_CONTEXT_TEMPLATE.format(
            today_utc=today_utc,
            tool_summary=tool_summary
        )

        # augmented: list[Message] = []
        # for i, msg in enumerate(original_messages):
        #     is_last = (i == len(original_messages) - 1)
        #     if is_last and msg.role == "user":
        #         augmented.append(
        #             Message(role=Role.SYSTEM, content=injection_content)
        #         )
        #     augmented.append(msg)
        # build_messages to get the properly layered message list,
        # injecting both RAG context and tool results via their respective slots:-
        from llm.prompt_builder import build_messages as _bm
        from memory.context_manager import get_context_manager

        # extract last user message for rebuild
        last_user = _last_user_message(original_messages)
        history: list[dict] = []
        summary: str | None = None
        # reconstruct history from original_messages (skip system and last user)
        i = 0
        msgs = list(original_messages)
        while i < len(msgs) - 1:
            if msgs[i].role == "user" and i + 1 < len(msgs) and msgs[i + 1].role == "assistant":
                history.append({"user": msgs[i].content, "assistant": msgs[i + 1].content})
                i += 2
            else:
                i += 1
        log.info("history&summary stream synthe", history=history, rag_context=rag_context)
        augmented = _bm(
            user_message=last_user,
            history_turns=history,
            summary=summary,
            rag_context=rag_context,
        )
        # append tool injection as an additional system message before the last user msg
        from models.schemas import Message as _Msg, Role as _Role
        final_user = augmented.pop() # remove the last USER message temporarily
        augmented.append(_Msg(role=_Role.SYSTEM, content=injection_content))
        augmented.append(final_user) # put it back last

        # Slightly lower temp: keeps persona voice, reduces factual invention
        # synthesis_temp = min(settings.default_temperature, 0.4)
        log.info("stream synthesize persona", final_prompt_to_llm=augmented)
        raw_stream = self.llm.stream(               # thinking params:-
            augmented,
            max_tokens=settings.default_max_tokens,
            temperature=0.6, #synthesis_temp,
            top_p=0.95,
            top_k=20,
            min_p=0,
            reasoning_effort="low"
        )
        async for token in raw_stream: # _strip_thinking(raw_stream):
            yield token

# non-streaming tool call

    async def _call_for_tools(
        self,
        messages: list[dict],
        tools: list[dict],
    ) -> dict[str, Any]:
        """
        Phase 1 call - stripped context, temp=0.0, tool_choice=auto.
        Returns the raw response dict including tool_calls.
        - Optionally, strips qwen3 <think> blocks from response content.
        """
        # the llm call with tool info. injected messages for retrieving tool_calls
        # extra_body: dict = {"repeat_penalty": settings.default_repeat_penalty, "think": True}
        extra_body = {}
        extra_body["top_k"] = 20
        extra_body["min_p"] = 0
        extra_body["reasoning_effort"] = "none"

        response = await self.llm._client.chat.completions.create(
            # model=settings.llm_model_name,
            # messages=messages,
            # tools=tools,
            # tool_choice="auto",
            # max_tokens=512,
            # temperature=0.0,    # deterministic tool selection
            # extra_body=extra_body,
            model=settings.llm_model_name,      # non-thinking params:-
            messages=messages,
            tools=tools,
            tool_choice="auto", 
            max_tokens=512,     # settings.default_max_tokens,
            temperature=0.7,    # 0.7 or 0.0 deterministic tool selection
            top_p=0.8,

            # temperature=0.6,    # thinking params. + set reasoning_effort to 'high'
            # top_p=0.95,
            # extra_body=extra_body,
        )
        log.info("callfortools_response", response)
        msg = response.choices[0].message
        result: dict[str, Any] = {
            "role": "assistant",
            "content": msg.content or "", # _strip_think_tags(msg.content or ""),
            "tool_calls": [],
        }
        if msg.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
        log.debug(
            "tool_llm_response",
            has_tool_calls=bool(result["tool_calls"]),
            tool_names=[tc["function"]["name"] for tc in result["tool_calls"]],
        )
        return result


# for Qwen3(.5) think-tag filtering - not using currently

# def _strip_think_tags(text: str) -> str:
#     """Removes all <think>...</think> blocks from a complete string."""
#     return _THINK_BLOCK_RE.sub("", text).strip()

# async def _strip_thinking(token_stream: AsyncIterator[str]) -> AsyncIterator[str]:
#     """
#     Filters <think>...</think> blocks from a live token stream.

#     Handles tags split across multiple tokens using a partial-match buffer:
#     if the end of the accumulated buffer looks like the start of a tag,
#     we hold it until the next token confirms or denies.
#     """
#     OPEN = "<think>"
#     CLOSE = "</think>"
#     buf = ""
#     inside = False

#     async for token in token_stream:
#         buf += token

#         while buf:
#             if inside:
#                 # drain until </think>
#                 idx = buf.find(CLOSE)
#                 if idx >= 0:
#                     buf = buf[idx + len(CLOSE):]
#                     inside = False
#                 else:
#                     keep = _partial_suffix(buf, CLOSE)
#                     buf = buf[-keep:] if keep else ""
#                     break
#             else:
#                 # scan for <think>
#                 idx = buf.find(OPEN)
#                 if idx >= 0:
#                     before = buf[:idx]
#                     if before:
#                         yield before
#                     buf = buf[idx + len(OPEN):]
#                     inside = True
#                 else:
#                     keep = _partial_suffix(buf, OPEN)
#                     if keep:
#                         safe = buf[:-keep]
#                         if safe:
#                             yield safe
#                         buf = buf[-keep:]
#                     else:
#                         yield buf
#                         buf = ""
#                     break

#     # flush remaining content if not inside a think block
#     if buf and not inside:
#         yield buf

# def _partial_suffix(text: str, tag: str) -> int:
#     """
#     Returns the length of the longest prefix of 'tag' that is a suffix of 'text'.
#     Used to hold a potential partial tag in the buffer across tokens.
#     """
#     for n in range(min(len(tag) - 1, len(text)), 0, -1):
#         if text.endswith(tag[:n]):
#             return n
#     return 0


# helper func.(s)

def _last_user_message(messages: list[Message]) -> str:
    """Returns the content of the last user-role message."""
    for msg in reversed(messages):
        if msg.role == "user":
            return msg.content
    return ""
def _last_user_messages(messages: list[Message], limit: int = 2) -> list[str]:
    """Returns the content of the last 'limit' user-role messages, newest last."""
    user_messages: list[str] = []

    for msg in reversed(messages):
        if msg.role == "user":
            user_messages.append(msg.content)
            if len(user_messages) == limit:
                break

    return list(reversed(user_messages))

def _format_tool_results(results: list) -> str:
    """
    Formats tool results into a clean summary for the synthesis prompt.
    The model reads this to compose its answer - clarity matters here.
    """
    parts: list[str] = []
    for r in results:
        if r.success:
            try:
                data = json.dumps(r.output, indent=2, default=str)
            except Exception:
                data = str(r.output)
            parts.append(f"[{r.tool_name}]\n{data}")
        else:
            parts.append(f"[{r.tool_name}] Failed: {r.error}")
    return "\n\n".join(parts) 


# singletons

_agent_loop: AgentLoop | None = None


def get_agent_loop() -> AgentLoop:
    if _agent_loop is None:
        raise RuntimeError("AgentLoop not initialised. Ensure MCP_ENABLED=true in .env")
    return _agent_loop


def init_agent_loop(
    llm_client,
    dispatcher: ToolDispatcher,
    registry: ToolRegistry,
    router: ToolRouter,
) -> AgentLoop:
    global _agent_loop
    _agent_loop = AgentLoop(llm_client, dispatcher, registry, router)
    log.info("agent_loop_init", max_iterations=_agent_loop.max_iterations)
    return _agent_loop
