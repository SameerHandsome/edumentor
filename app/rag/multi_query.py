"""
Multi-query transform for user_memory retrieval.

Generates N alternative phrasings of the original query, embeds each,
then merges the result sets (deduped by point id, highest score wins).

Used exclusively for user_memory retrieval — HyDE is used for curriculum_docs.
"""

from __future__ import annotations

import httpx
import structlog

from app.core.config import settings
from app.rag.embeddings import embed_batch

logger = structlog.get_logger(__name__)

_OLLAMA_GENERATE_URL = f"{settings.OLLAMA_BASE_URL}/api/generate"
_NUM_VARIANTS = 3  # how many alternative queries to generate

_MULTI_QUERY_PROMPT = """\
You are a search assistant. Given the original query below, generate {n} \
alternative phrasings that capture the same intent using different vocabulary. \
Output ONLY the alternatives, one per line, no numbering, no extra text.

Original query: {query}

Alternatives:"""


async def multi_query_transform(query: str) -> list[list[float]]:
    """
    Return a list of dense vectors:
      - index 0  : embedding of the original query (always included)
      - indices 1…N : embeddings of LLM-generated query variants

    Callers search with each vector and union the result sets.
    Falls back to [original_vector] if LLM generation fails.
    """
    variants = await _generate_variants(query)
    all_queries = [query] + variants
    logger.debug("multi_query_variants", count=len(all_queries))
    return await embed_batch(all_queries)


async def _generate_variants(query: str) -> list[str]:
    """Ask Ollama to produce alternative phrasings of the query."""
    prompt = _MULTI_QUERY_PROMPT.format(n=_NUM_VARIANTS, query=query)
    try:
        async with httpx.AsyncClient(timeout=float(settings.OLLAMA_TIMEOUT_SECONDS)) as client:
            resp = await client.post(
                _OLLAMA_GENERATE_URL,
                json={
                    "model": settings.OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.5, "num_predict": 120},
                },
            )
            resp.raise_for_status()
            raw: str = resp.json().get("response", "")
            lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
            # Keep at most _NUM_VARIANTS and discard empty / near-duplicate lines
            return lines[:_NUM_VARIANTS]
    except Exception as exc:
        logger.error("multi_query_generation_error", error=str(exc))
        return []
