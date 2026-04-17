"""
the async LLM client - Network layer 

wraps the OpenAI-compatible REST API exposed by 'llama-server',
supports both streaming (SSE) and non-streaming completion,
handles retries, timeouts, and some sweet logging

based on the *Adapter Pattern*, allowing us to swap out the underlying LLM server with minimal changes to the rest of the codebase.
"""

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any

# from fastapi import params
import httpx
from openai import AsyncOpenAI, APIConnectionError, APIStatusError, APITimeoutError

from config import get_settings
from models.schemas import Message
from llm.prompt_builder import messages_to_dicts
from utils.logger import get_logger

log = get_logger(__name__)
settings = get_settings()


class LLMClient:
    """
    a thin async wrapper around the OpenAI-compatible llama.cpp server

    Retry policy: 
        - exponential back-off on connection/timeout errors.
        - Mid-stream errors are NOT retried (partial output already delivered).

    Usage::

        client = LLMClient()
        await client.health_check()

        # streaming
        async for token in client.stream("Hello!", history=[...]):
            print(token, end="", flush=True)

        # non-streaming
        response = await client.complete("Hello!", history=[...])
    """

    def __init__(self) -> None:
        self._client = AsyncOpenAI(
            base_url=f"{settings.llm_base_url}/v1",
            api_key=settings.llm_api_key,
            timeout=httpx.Timeout(
                connect=10.0,
                read=settings.llm_request_timeout,
                write=10.0,
                pool=5.0,
            ),
            max_retries=0,  # we handle retries ourselves for finer control
        )
        log.info(
            "llm_client_init",
            base_url=settings.llm_base_url,
            model=settings.llm_model_name,
        )

    async def stream(
        self,
        messages: list[Message],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        min_p: int | None = None,
        reasoning_effort: str | None = None,
        # think: bool | None = None,
    ) -> AsyncIterator[str]:
        """
        a generator to 'yield' individual token strings as they arrive from the model,
        handles reconn./retry up to 'settings.llm_max_retries' times

        Retries initial connection failures with exponential back-off.
        Does not retry mid-stream errors (user already received partial text).
        """
        params = self._build_params(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            stream=True,
            reasoning_effort=reasoning_effort
        )
        attempt = 0
        while attempt <= settings.llm_max_retries:
            try:
                t0 = time.perf_counter()
                # token_count = 0
                # content_parts = []
                # reasoning_parts = []
                content_tokens = 0
                reasoning_tokens = 0
                finish_reason = None
                log.info("params_for_stream", params=params)
                stream = await self._client.chat.completions.create(**params)
                async for chunk in stream:
                    log.info("chunk", chunk)
                    if not chunk.choices:
                        continue

                    choice = chunk.choices[0]
                    finish_reason = choice.finish_reason or finish_reason
                    delta = choice.delta

                    # Reasoning field (Ollama thinking models), delta.reasoning contains chain-of-thought tokens,
                    # We do NOT yield these, they are internal model thought,
                    # Access via getattr because it's a non-standard extension field not present in the OpenAI SDK's typed ChoiceDelta
                    reasoning_token = getattr(delta, "reasoning", None)
                    if reasoning_token:
                        reasoning_tokens += 1
                        # continue # skip - not emitted to callers
                        # todo: yield for preview thinking process !

                    # # has_finished_reason = choice.finish_reason
                    # text = getattr(delta, "content", None)
                    # if text:
                    #     token_count += 1
                    #     yield text
                    #     # log.info("yielded text", text)
                    # else:
                    #     log.debug("llm_stream_empty_chunk", chunk=chunk.model_dump())
                    # content = getattr(delta, "content", None)
                    # reasoning = getattr(delta, "reasoning", None) # or getattr(delta, "thinking", None)

                    # content field (actual response)
                    if delta.content:
                        content_tokens += 1
                        yield delta.content

                elapsed = time.perf_counter() - t0

                # on token budget exhaustion..
                if finish_reason == "length":
                    if content_tokens == 0 and reasoning_tokens > 0:
                        log.warning(
                            "llm_stream_no_content",
                            reason=(
                                "Token budget exhausted during reasoning phase - "
                                "zero content tokens produced. "
                                f"Increase default_max_tokens (current: "
                                f"{params['max_tokens']}) in .env."
                            ),
                            reasoning_tokens=reasoning_tokens,
                            max_tokens=params["max_tokens"],
                        )
                        yield "outta tokens.."
                    else:
                        log.warning(
                            "llm_stream_truncated",
                            reason="Response truncated by max_tokens limit",
                            content_tokens=content_tokens,
                            max_tokens=params["max_tokens"],
                        )

                if finish_reason == "stop":
                    if content_tokens == 0 and reasoning_tokens > 0:
                        log.warning(
                            "llm_stream_no_content",
                            reason=(
                                "finish reason 'stop' reached"
                            ),
                            reasoning_tokens=reasoning_tokens,
                            max_tokens=params["max_tokens"],
                        )
                        yield "finish reason 'stop' reached, model only reasoned."
                    else:
                        log.warning(
                            "llm_stream_truncated",
                            reason="Response truncated by max_tokens limit w/ finish_reason='stop'",
                            content_tokens=content_tokens,
                            max_tokens=params["max_tokens"],
                        )


                log.info(
                    "llm_stream_complete",
                    content_tokens=content_tokens,
                    reasoning_tokens=reasoning_tokens,
                    finish_reason=finish_reason,
                    elapsed_s=round(elapsed, 3),
                    tok_per_sec=round(content_tokens / elapsed, 1) if elapsed and content_tokens else 0,
                )
                return # success, end the async generator

            except (APIConnectionError, APITimeoutError) as exc:
                attempt += 1
                log.warning(
                    "llm_stream_retry",
                    attempt=attempt,
                    max=settings.llm_max_retries,
                    error=str(exc),
                )
                if attempt > settings.llm_max_retries:
                    log.error("llm_stream_failed", error=str(exc))
                    raise
                # if llama server is momentarily busy, wait progressively longer between retries rather than hammering it
                await asyncio.sleep(2 ** attempt) # 2s, 4s, 8s...

            except APIStatusError as exc:
                log.error("llm_api_error", status=exc.status_code, body=exc.message)
                raise

    async def complete(
        self,
        messages: list[Message],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        min_p: int | None = None,
        reasoning_effort: str | None = None,
        # think: bool | None = None,
    ) -> tuple[str, dict[str, int]]:
        
        """
        non-streaming completion
        Returns:
            (response_text, usage_dict)
        """
        params = self._build_params(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            stream=False,
            reasoning_effort=reasoning_effort,
            # think=think, reasoning_effort="low",
        )
        attempt = 0
        while True:
            try:
                t0 = time.perf_counter()
                log.info("params_for_complete", params=params)
                response = await self._client.chat.completions.create(**params)
                elapsed = time.perf_counter() - t0

                #text = response.choices[0].message.content or ""
                log.info("response", response)
                choice = response.choices[0]
                msg = choice.message
                text = msg.content or ""

                # For thinking models in non-streaming mode, content may be empty
                # if the model exhausted its token budget on reasoning
                if not text:
                    reasoning = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None)
                    if reasoning:
                        log.warning(
                            "llm_complete_no_content",
                            reason="Token budget exhausted during reasoning - content is empty",
                            reasoning_len=len(reasoning),
                            max_tokens=params["max_tokens"],
                        )
                    return reasoning or "", {}

                finish_reason = response.choices[0].finish_reason
                if finish_reason == "length":
                    log.warning("llm_complete_truncated", max_tokens=params["max_tokens"])

                usage: dict[str, int] = {}
                if response.usage:
                    usage = {
                        "prompt_tokens":     response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens":      response.usage.total_tokens,
                    }
                log.info("llm_complete", elapsed_s=round(elapsed, 3), usage=usage, finish_reason=finish_reason)
                return text, usage

            except (APIConnectionError, APITimeoutError) as exc:
                attempt += 1
                log.warning("llm_complete_retry", attempt=attempt, error=str(exc))
                if attempt > settings.llm_max_retries:
                    raise
                await asyncio.sleep(2 ** attempt)

            except APIStatusError as exc:
                log.error("llm_api_error", status=exc.status_code, body=exc.message)
                raise

    async def health_check(self) -> tuple[bool, float]:
        """
        for llama-server health endpoint
        Returns:
            (is_healthy, latency_ms)
        """
        try:
            t0 = time.perf_counter() 
            async with httpx.AsyncClient(timeout=5.0) as http:
                # r = await http.get(f"{settings.llm_base_url}/health")
                r = await http.get(f"{settings.llm_base_url}")  
            latency = (time.perf_counter() - t0) * 1000
            healthy = r.status_code == 200
            log.debug("llm_health", status=r.status_code, latency_ms=round(latency, 1))
            return healthy, latency
        except Exception as exc:
            log.warning("llm_health_fail", error=str(exc))
            return False, 0.0

    def _build_params(
        self,
        messages: list[Message],
        *,
        max_tokens: int | None,
        temperature: float | None,
        top_p: float | None,
        top_k: int | None,
        min_p: int | None,
        stream: bool,
        # think: bool | None = None,
        reasoning_effort: str = "low",
    ) -> dict[str, Any]:
        # extra: dict[str, Any] = {"repeat_penalty": settings.default_repeat_penalty, "reasoning_effort": reasoning_effort}
        # extra_body : dict[dict]= {
        #     "options": {
        #         "top_k": top_k if top_k is not None else settings.default_top_k,
        #         "min_p": min_p if min_p is not None else settings.default_min_p,
        #     }
        # }
        # # Ollama-specific: {"think": false} disables the reasoning phase,
        # # Safe to include with non-Ollama backends - unrecognised extra_body
        # # keys are ignored by most OpenAI-compatible servers
        # if think is not None:
        #     extra["think"] = think

        params = {
            "model":       settings.llm_model_name,
            "messages":    messages_to_dicts(messages),
            "max_tokens":  max_tokens if max_tokens is not None else settings.default_max_tokens,
            #"presence_penalty": 1.5,
            "temperature": temperature if temperature is not None else settings.default_temperature,
            #"top_k":       top_k if top_k is not None else settings.default_top_k,
            "top_p":       top_p if top_p is not None else settings.default_top_p,
            #"min_p":       min_p if min_p is not None else settings.default_min_p,
            "stream":      stream,
            #"reasoning_effort": reasoning_effort,
            #"extra_body":  extra_body,
        }
        extra_body = {}
        if top_k is not None:
            extra_body["top_k"] = top_k
        if min_p is not None:
            extra_body["min_p"] = min_p
        extra_body["reasoning_effort"] = reasoning_effort
        
        if extra_body:
            params["extra_body"] = extra_body
        

        return params
            
            
    async def aclose(self) -> None:
        await self._client.close()
        


# coz there should be exactly one LLM client for the entire application, we define some singletons here,
# creating a new HTTP connection pool per request would be wasteful and slow,
# so a singleton is created once at startup and shared via FastAPI dependency ('get_llm_client')
_llm_client: LLMClient | None = None

def get_llm_client() -> LLMClient:
    """FastAPI dependency: returns the module-level LLM client."""
    if _llm_client is None:
        raise RuntimeError("LLMClient not initialised. Check lifespan setup.")
    return _llm_client

def init_llm_client() -> LLMClient:
    global _llm_client
    _llm_client = LLMClient()
    return _llm_client

def shutdown_llm_client() -> None:
    global _llm_client
    _llm_client = None
