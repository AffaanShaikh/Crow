"""
Local embedding generation via Ollama, uses Ollama's OpenAI-compatible /v1/embeddings endpoint.
No external API calls - everything runs on your machine!

Default model we're using: nomic-embed-text
  - 768 dimensions
  - 8192 context window
  - Fast, good quality, widely supported
  - Pull with: ollama pull nomic-embed-text

Alternative: mxbai-embed-large (1024 dims, higher quality, slower)
             all-minilm (384 dims, fastest, lower quality)

Batching:-
    Ollama processes one embedding at a time efficiently. We batch our calls
    using asyncio.gather() to parallelise multiple embeddings.
    Batch size is configurable - larger batches use more VRAM simultaneously.
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from config import get_settings
from utils.logger import get_logger

log = get_logger(__name__)
settings = get_settings()


class EmbeddingClient:
    """
    Async client for generating text embeddings via Ollama.

    Usage::

        embedder = EmbeddingClient()
        vector = await embedder.embed("Hello world")
        vectors = await embedder.embed_many(["text1", "text2", "text3"])
    """

    def __init__(self, model: str | None = None) -> None:
        self.model = model or settings.rag_embedding_model
        self._base_url = settings.llm_base_url # same Ollama instance

    async def embed(self, text: str) -> list[float]:
        """
        Embed a single string. Returns a list of floats (the vector).
        Raises RuntimeError if Ollama is unavailable or the model is not pulled.
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text")

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{self._base_url}/v1/embeddings",
                json={"model": self.model, "input": text},
            )

        if resp.status_code == 404:
            raise RuntimeError(
                f"Embedding model '{self.model}' not found. "
                f"Run: ollama pull {self.model}"
            )
        if not resp.is_success:
            raise RuntimeError(
                f"Ollama embedding failed: HTTP {resp.status_code} - {resp.text[:200]}"
            )

        data = resp.json()
        vector = data["data"][0]["embedding"]
        log.debug("embedded", model=self.model, dims=len(vector), chars=len(text))
        return vector

    async def embed_many(
        self,
        texts: list[str],
        batch_size: int = 16,
    ) -> list[list[float]]:
        """
        Embed multiple strings. Returns vectors in the same order as input.
        Processes in batches to avoid overwhelming Ollama.
        """
        if not texts:
            return []

        results: list[list[float] | None] = [None] * len(texts)

        for batch_start in range(0, len(texts), batch_size):
            batch = texts[batch_start : batch_start + batch_size]
            batch_vectors = await asyncio.gather(
                *[self.embed(t) for t in batch],
                return_exceptions=False,
            )
            for i, vec in enumerate(batch_vectors):
                results[batch_start + i] = vec

            if batch_start + batch_size < len(texts):
                # a small pause between batches to avoid saturating Ollama
                await asyncio.sleep(0.05)

        log.info(
            "embed_batch_complete",
            model=self.model,
            total=len(texts),
            dims=len(results[0]) if results else 0,
        )
        return results # type: ignore[return-value]

    async def health_check(self) -> tuple[bool, str]:
        """
        Verify the embedding model is available.
        Returns (is_ready, message).
        """
        try:
            await self.embed("ping")
            return True, f"{self.model} ready"
        except RuntimeError as exc:
            return False, str(exc)
        except Exception as exc:
            return False, f"Unreachable: {exc}"


# module-level singleton

_embedder: EmbeddingClient | None = None


def get_embedder() -> EmbeddingClient:
    if _embedder is None:
        raise RuntimeError("EmbeddingClient not initialised.")
    return _embedder


def init_embedder() -> EmbeddingClient:
    global _embedder
    _embedder = EmbeddingClient()
    log.info("embedder_init", model=settings.rag_embedding_model)
    return _embedder