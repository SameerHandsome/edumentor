"""
CRAG — Corrective RAG.
...
"""

from __future__ import annotations

import os

import structlog
from tavily import AsyncTavilyClient

from app.rag.reranker import RankedResult
from app.rag.retriever import retrieve_curriculum

logger = structlog.get_logger(__name__)

_HIGH_THRESHOLD = 5.0  # cross-encoder logit: good match (ms-marco-MiniLM typical range 5–8)
_LOW_THRESHOLD = 0.0  # cross-encoder logit: below this → treat as no good local match


async def crag_retrieve(
    client,
    bm25_encoder,
    query: str,
    *,
    top_k: int = 3,
    topic_id: str | None = None,
) -> tuple[list[RankedResult], str]:
    # retrieve_curriculum already calls rerank() internally and returns List[RankedResult].
    # Do NOT call rerank() again — passing NamedTuples to rerank() causes
    # "tuple indices must be integers or slices, not str"
    results = await retrieve_curriculum(
        client, bm25_encoder, query, topic_id=topic_id, top_k=top_k * 2
    )
    if not results:
        return [], "empty"

    top_score = results[0].score

    if top_score > _HIGH_THRESHOLD:
        logger.info("crag_correct_using_as_is", score=top_score)
        return results[:top_k], "correct"

    if top_score > _LOW_THRESHOLD:
        logger.info("crag_ambiguous_supplementing", score=top_score)
        return results[:top_k], "ambiguous"

    logger.warning("crag_incorrect_searching_web", score=top_score, query=query[:60])
    refined_query = await _refine_query(query)

    try:
        tavily_api_key = os.environ.get("TAVILY_API_KEY")
        if not tavily_api_key:
            logger.error("TAVILY_API_KEY is missing. Falling back to local db.")
            return results[:top_k], "fallback"

        tavily_client = AsyncTavilyClient(api_key=tavily_api_key)
        search_result = await tavily_client.search(
            query=refined_query, search_depth="basic", max_results=3
        )

        corrected_results = []
        for i, res in enumerate(search_result.get("results", [])):
            corrected_results.append(
                RankedResult(
                    payload={
                        "content": f"WEB SEARCH [{res['title']}]: {res['content']}",
                        "source": res["url"],
                    },
                    score=1.0,
                    point_id=f"web_result_{i}",
                )
            )

        if corrected_results:
            logger.info("crag_tavily_search_succeeded", refined_query=refined_query[:60])
            return corrected_results, "web_corrected"

    except Exception as e:
        logger.error("tavily_search_failed", error=str(e))

    return results[:top_k], "fallback"


async def _refine_query(query: str) -> str:
    from app.core.model_router import routed_chat

    messages = [
        {
            "role": "system",
            "content": (
                "You are a search query optimizer. Given a question, rewrite it "
                "to be more specific and searchable for a web search engine. Focus on key concepts. "
                "Return ONLY the rewritten query, no explanation."
            ),
        },
        {"role": "user", "content": f"Original query: {query}\n\nRewritten query:"},
    ]
    refined = await routed_chat(messages, agent_type="default", temperature=0.2, max_tokens=60)
    return refined.strip()
