"""
Self-RAG — Selective Retrieval with Reflection Tokens.

Self-RAG makes retrieval conditional:
  1. Retrieve? (ISREL token) — should we even retrieve for this query?
  2. Supported? (ISSUP token) — does the retrieved passage support the generation?
  3. Useful? (ISUSE token)   — is the final response useful to the user?

Simplified 3-token implementation using our Phi-3.5 model as the critic.
Used as a post-generation quality gate before returning to the user.

Reference: "Self-RAG: Learning to Retrieve, Generate, and Critique" (Asai et al., 2023)
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog

from app.core.model_router import routed_chat

logger = structlog.get_logger(__name__)


@dataclass
class SelfRAGDecision:
    should_retrieve: bool
    is_supported: bool
    is_useful: bool
    confidence: float  # 0.0–1.0 aggregate
    critique: str  # short explanation from the critic


async def assess_retrieval_need(query: str) -> bool:
    """
    ISREL: Decide if retrieval is needed for this query.
    Simple heuristics first (fast path), then LLM critic.
    """
    # Fast path: pure social/acknowledgement queries don't need retrieval.
    # Matched by exact normalized form only — NOT by word count.
    #
    # ⚠️  Do NOT add a word-count guard here (e.g. len < 3).
    # Short queries like "text classification", "neural networks", "back
    # propagation", "NLP", "DNA" are genuine educational topic requests that
    # must go through CRAG.  The LLM critic below handles ambiguous cases.
    low_info_patterns = {
        "hi",
        "hello",
        "thanks",
        "thank you",
        "ok",
        "okay",
        "yes",
        "no",
        "sure",
        "got it",
        "i see",
        "understood",
        "bye",
        "goodbye",
        "great",
        "nice",
        "cool",
        "wow",
    }
    normalized = query.lower().strip().rstrip("!?.")
    if normalized in low_info_patterns:
        logger.debug("self_rag_fast_path_skip", query=query[:60])
        return False

    # LLM critic for ambiguous cases
    messages = [
        {
            "role": "system",
            "content": (
                "You decide if a student's query to a tutoring AI needs factual "
                "reference material to answer well. Short topic names like "
                "'text classification', 'NLP', 'DNA', 'backpropagation' DO need "
                "retrieval. Pure greetings or acknowledgements like 'ok', 'thanks', "
                "'hi' do NOT. Reply with only YES or NO."
            ),
        },
        {"role": "user", "content": f"Query: {query}"},
    ]
    try:
        verdict = await routed_chat(messages, agent_type="intent", temperature=0.0, max_tokens=5)
        verdict = (verdict or "").strip()

        # If the model returned empty or something that contains neither YES nor NO
        # (e.g. rate-limit message, apology, empty string) — default to retrieve.
        if not verdict or ("YES" not in verdict.upper() and "NO" not in verdict.upper()):
            logger.warning(
                "self_rag_unexpected_verdict_defaulting_to_retrieve",
                verdict=repr(verdict),
                query=query[:60],
            )
            return True

        result = "YES" in verdict.upper()
        logger.info("self_rag_retrieval_decision", verdict=verdict, result=result, query=query[:60])
        return result
    except Exception as exc:
        logger.warning(
            "self_rag_critic_failed_defaulting_to_retrieve", error=str(exc), query=query[:60]
        )
        return True  # always retrieve if the critic fails


async def critique_response(
    query: str,
    response: str,
    retrieved_chunks: list[str],
) -> SelfRAGDecision:
    """
    ISSUP + ISUSE: Evaluate whether the generated response is:
    1. Supported by the retrieved passages (no hallucination)
    2. Actually useful for the student

    Returns a SelfRAGDecision with scores and a critique string.
    """

    context = "\n---\n".join(retrieved_chunks[:2]) if retrieved_chunks else "No context retrieved."

    messages = [
        {
            "role": "system",
            "content": (
                "You are a strict QA critic for an educational AI. "
                "Evaluate the RESPONSE against the CONTEXT and QUESTION.\n"
                "Reply in this exact format:\n"
                "SUPPORTED: YES/NO\n"
                "USEFUL: YES/NO\n"
                "CRITIQUE: one sentence"
            ),
        },
        {
            "role": "user",
            "content": (
                f"QUESTION: {query}\n\n" f"CONTEXT:\n{context}\n\n" f"RESPONSE: {response}"
            ),
        },
    ]
    raw = await routed_chat(messages, agent_type="default", temperature=0.1, max_tokens=80)

    # Parse structured response
    is_supported = "SUPPORTED: YES" in raw.upper()
    is_useful = "USEFUL: YES" in raw.upper()
    critique = ""
    for line in raw.splitlines():
        if line.upper().startswith("CRITIQUE:"):
            critique = line.split(":", 1)[-1].strip()
            break

    confidence = (0.5 if is_supported else 0.0) + (0.5 if is_useful else 0.0)

    logger.info(
        "self_rag_critique",
        supported=is_supported,
        useful=is_useful,
        confidence=confidence,
        query=query[:50],
    )
    return SelfRAGDecision(
        should_retrieve=True,
        is_supported=is_supported,
        is_useful=is_useful,
        confidence=confidence,
        critique=critique,
    )
