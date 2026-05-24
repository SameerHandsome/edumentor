"""
Embedding service — nomic-embed-text (768-dim) via Ollama.
Handles single texts and batches with retry logic.

Ollama Embed API (current):
  POST /api/embed
  Body : { "model": "...", "input": "text" | ["text1", "text2"] }
  Response: { "embeddings": [[...], [...]], ... }

NOTE: The old /api/embeddings endpoint with "prompt" / "embedding" keys
is DEPRECATED and removed in recent Ollama versions. Always use /api/embed.
Ollama's /api/embed natively accepts a list for "input", so we use that
for batches instead of looping — faster and fewer HTTP round trips.
"""

from __future__ import annotations

import asyncio

import httpx
import structlog

from app.core.config import settings

logger = structlog.get_logger(__name__)

# Current Ollama embed endpoint — NOT /api/embeddings (that is the old path)
_EMBED_URL = f"{settings.OLLAMA_BASE_URL}/api/embed"
_MAX_RETRIES = 3
_RETRY_DELAY = 1.0  # seconds, multiplied by attempt number for backoff


async def embed_text(text: str) -> list[float]:
    """
    Embed a single string. Returns a 768-dim float list.
    Retries up to _MAX_RETRIES times on transient errors.
    """
    results = await embed_batch([text])
    return results[0]


async def embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Embed multiple texts in one HTTP call.
    Ollama /api/embed accepts a list in the "input" field and returns
    "embeddings" as a list of vectors in the same order.
    Returns list of 768-dim vectors.
    """
    if not texts:
        return []
    async with httpx.AsyncClient(timeout=float(settings.OLLAMA_TIMEOUT_SECONDS)) as client:
        return await _embed_batch_with_retry(client, texts)


async def _embed_batch_with_retry(
    client: httpx.AsyncClient,
    texts: list[str],
    attempt: int = 0,
) -> list[list[float]]:
    """Inner retry loop for a batch embedding request."""
    try:
        resp = await client.post(
            _EMBED_URL,
            json={
                "model": settings.OLLAMA_EMBED_MODEL,
                # "input" accepts a single string OR a list of strings
                "input": texts,
            },
        )
        resp.raise_for_status()
        data = resp.json()

        # Response key is "embeddings" (list of lists) — NOT "embedding"
        embeddings: list[list[float]] = data["embeddings"]

        if len(embeddings) != len(texts):
            raise ValueError(
                f"Ollama returned {len(embeddings)} embeddings for " f"{len(texts)} inputs"
            )
        for i, vec in enumerate(embeddings):
            if len(vec) != 768:
                raise ValueError(f"Expected 768-dim vector at index {i}, got {len(vec)}")
        return embeddings

    except (ValueError, KeyError) as exc:
        # Non-transient errors — wrong dimensions or missing response key.
        # Retrying cannot fix these; raise immediately.
        logger.error("embedding_bad_response", texts_count=len(texts), error=str(exc))
        raise
    except (httpx.HTTPStatusError, httpx.RequestError) as exc:
        # Transient network / server errors — safe to retry with backoff.
        if attempt < _MAX_RETRIES - 1:
            logger.warning(
                "embedding_retry",
                attempt=attempt + 1,
                texts_count=len(texts),
                error=str(exc),
            )
            await asyncio.sleep(_RETRY_DELAY * (attempt + 1))
            return await _embed_batch_with_retry(client, texts, attempt + 1)
        logger.error("embedding_failed", texts_count=len(texts), error=str(exc))
        raise
