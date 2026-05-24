"""
Shadow Testing (Shadow Mode) for model evaluation.

Shadow testing runs a NEW (candidate) model in parallel with the LIVE model,
but only the live model's response is returned to the user.
The candidate's response is logged to LangSmith for offline comparison.

Use case:
  Before promoting a newly fine-tuned Phi-3.5 to production,
  run it in shadow mode for 1000 requests to compare quality.

Activation: set SHADOW_MODEL_ENABLED=true and SHADOW_MODEL_NAME=<model> in .env
"""

from __future__ import annotations

import asyncio

import structlog
from prometheus_client import Counter

logger = structlog.get_logger(__name__)

SHADOW_REQUESTS = Counter("shadow_test_requests_total", "Total shadow test requests fired")
SHADOW_ERRORS = Counter("shadow_test_errors_total", "Shadow test failures (non-blocking)")


async def shadow_call(
    messages: list[dict],
    live_response: str,
    *,
    agent_type: str = "default",
    session_id: str = "",
    temperature: float = 0.7,
    max_tokens: int = 512,
) -> None:
    """
    Fire-and-forget: call shadow model and log comparison to LangSmith.
    Never blocks the live request. Errors are swallowed and counted.
    """
    from app.core.config import settings

    shadow_enabled = getattr(settings, "SHADOW_MODEL_ENABLED", False)
    shadow_model = getattr(settings, "SHADOW_MODEL_NAME", "")
    if not shadow_enabled or not shadow_model:
        return

    asyncio.create_task(
        _run_shadow(
            messages,
            live_response,
            shadow_model=shadow_model,
            agent_type=agent_type,
            session_id=session_id,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    )


async def _run_shadow(
    messages: list[dict],
    live_response: str,
    *,
    shadow_model: str,
    agent_type: str,
    session_id: str,
    temperature: float,
    max_tokens: int,
) -> None:
    """
    Actual shadow execution. Runs concurrently, result only goes to LangSmith.
    """
    import httpx

    from app.core.config import settings

    SHADOW_REQUESTS.inc()
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{settings.OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": shadow_model,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": temperature, "num_predict": max_tokens},
                },
            )
            resp.raise_for_status()
            shadow_response = resp.json()["message"]["content"].strip()

        # Log comparison to LangSmith
        from langsmith import Client

        if settings.LANGSMITH_API_KEY:
            ls_client = Client()
            ls_client.create_example(
                inputs={"messages": messages, "agent_type": agent_type, "session_id": session_id},
                outputs={
                    "live_response": live_response,
                    "shadow_response": shadow_response,
                    "live_model": settings.OLLAMA_MODEL,
                    "shadow_model": shadow_model,
                },
                dataset_name="shadow_test_comparisons",
            )
        logger.debug("shadow_logged", agent_type=agent_type, session_id=session_id)
    except Exception as exc:
        SHADOW_ERRORS.inc()
        logger.warning("shadow_test_failed", error=str(exc))
