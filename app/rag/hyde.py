"""
HyDE (Hypothetical Document Embeddings) query transform.

Instead of embedding the raw query, we ask the LLM to generate a
hypothetical document that would answer the query, then embed THAT.
The hypothesis lives in the same semantic space as real curriculum
chunks — dramatically improving recall for short/ambiguous questions.

Used exclusively for curriculum_docs retrieval.
For user_memory we use multi-query (see query_transforms.py).
"""

from __future__ import annotations

import httpx
import structlog

from app.core.config import settings
from app.rag.embeddings import embed_text

logger = structlog.get_logger(__name__)

_OLLAMA_CHAT_URL = f"{settings.OLLAMA_BASE_URL}/api/chat"

_HYDE_SYSTEM = (
    "You are a curriculum author. Given a student's question, write a "
    "concise, factually accurate paragraph (3–5 sentences) that directly "
    "answers the question as if from a textbook. "
    "Return ONLY the paragraph — no headings, no preamble, no commentary."
)


async def hyde_transform(query: str) -> list[float]:
    """
    1. Generate a hypothetical textbook passage for `query` via Ollama.
    2. Embed the passage with nomic-embed-text.
    3. Return the dense vector.

    Falls back to embedding the raw query if generation fails.
    """
    hypothesis = await _generate_hypothesis(query)
    if not hypothesis:
        logger.warning("hyde_fallback_raw_query", query=query[:80])
        return await embed_text(query)

    logger.debug("hyde_hypothesis", length=len(hypothesis))
    return await embed_text(hypothesis)


async def _generate_hypothesis(query: str) -> str:
    """
    Call Ollama's chat endpoint with the Phi-3.5 model to produce
    the hypothetical document. Uses /api/chat with proper system/user
    roles for better instruction following than /api/generate with a
    manually concatenated prompt.
    """
    try:
        async with httpx.AsyncClient(timeout=float(settings.OLLAMA_TIMEOUT_SECONDS)) as client:
            resp = await client.post(
                _OLLAMA_CHAT_URL,
                json={
                    "model": settings.OLLAMA_MODEL,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 200,
                    },
                    "messages": [
                        {"role": "system", "content": _HYDE_SYSTEM},
                        {
                            "role": "user",
                            "content": (f"Student question: {query}\n\nTextbook passage:"),
                        },
                    ],
                },
            )
            resp.raise_for_status()
            data = resp.json()
            # /api/chat response: {"message": {"role": "assistant", "content": "..."}}
            return data.get("message", {}).get("content", "").strip()
    except Exception as exc:
        logger.error("hyde_generation_error", error=str(exc))
        return ""
