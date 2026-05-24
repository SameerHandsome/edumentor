"""
Shared ChatOllama client factory.
Routes through the cascade model router (CB + fallback).
Also fires shadow calls when shadow testing is enabled.
"""

from __future__ import annotations

import structlog
from langchain_ollama import ChatOllama

from app.core.config import settings

logger = structlog.get_logger(__name__)


def get_chat_ollama(
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 512,
) -> ChatOllama:
    """Return a ChatOllama instance for the given model."""
    return ChatOllama(
        model=model or settings.OLLAMA_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=temperature,
        num_predict=max_tokens,
    )


async def ollama_chat(
    messages: list[dict],
    temperature: float = 0.7,
    max_tokens: int = 512,
    agent_type: str = "default",
    session_id: str = "",
) -> str:
    """
    Send messages through the cascade router + optional shadow testing.
    Keeps backward compatibility with agent nodes that call ollama_chat directly.
    """
    from app.core.model_router import routed_chat_with_shadow

    reply = await routed_chat_with_shadow(
        messages,
        agent_type=agent_type,
        temperature=temperature,
        max_tokens=max_tokens,
        session_id=session_id,
    )
    logger.debug("ollama_chat_reply", chars=len(reply), agent_type=agent_type)
    return reply
