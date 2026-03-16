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

from fastapi import params
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
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """
        a generator to 'yield' individual token strings as they arrive from the model,
        handles reconn./retry up to 'settings.llm_max_retries' times
        """
        params = self._build_params(
            messages, temperature=temperature, max_tokens=max_tokens, stream=True
        )
        attempt = 0
        while attempt <= settings.llm_max_retries:
            try:
                t0 = time.perf_counter()
                token_count = 0

                # async with self._client.chat.completions.create(**params) as stream:
                stream = await self._client.chat.completions.create(**params)
                async for chunk in stream:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        token_count += 1
                        yield delta

                elapsed = time.perf_counter() - t0
                log.info(
                    "llm_stream_complete",
                    tokens=token_count,
                    elapsed_s=round(elapsed, 3),
                    tok_per_sec=round(token_count / elapsed, 1) if elapsed else 0,
                )
                return  # success

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
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> tuple[str, dict[str, int]]:
        """
        non-streaming completion
        Returns:
            (response_text, usage_dict)
        """
        params = self._build_params(
            messages, temperature=temperature, max_tokens=max_tokens, stream=False
        )
        attempt = 0
        while True:
            try:
                t0 = time.perf_counter()
                response = await self._client.chat.completions.create(**params)
                elapsed = time.perf_counter() - t0
                text = response.choices[0].message.content or ""
                usage: dict[str, int] = {}
                if response.usage:
                    usage = {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                    }
                log.info(
                    "llm_complete",
                    elapsed_s=round(elapsed, 3),
                    usage=usage,
                )
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
        temperature: float | None,
        max_tokens: int | None,
        stream: bool,
    ) -> dict[str, Any]:
        return {
            "model": settings.llm_model_name,
            "messages": messages_to_dicts(messages),
            "temperature": temperature if temperature is not None else settings.default_temperature,
            "max_tokens": max_tokens if max_tokens is not None else settings.default_max_tokens,
            "top_p": settings.default_top_p,
            "stream": stream,
            # llama.cpp extra params (ignored by real OpenAI but honoured locally)
            "extra_body": {"repeat_penalty": settings.default_repeat_penalty},
        }

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
