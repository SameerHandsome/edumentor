"""
Cascade Model Router — 3-tier LLM routing with circuit breakers.

Tier 1 (Primary):   Fine-tuned Phi-3.5-mini (edumentor-phi3.5) via Ollama
Tier 2 (Secondary): Fallback model (llama3.1:8b or any installed Ollama model)
Tier 3 (Stub):      Minimal hard-coded graceful-degradation response

Routing logic:
  1. Check primary circuit breaker → try primary model
  2. If primary CB is OPEN or call fails → check secondary CB → try secondary
  3. If both CBs are OPEN → return stub response (never crashes the request)

This implements the "Cascade" pattern from the ML routing taxonomy.
The router is also where Mixture-of-Agents (MoA) can be plugged in later:
instead of tier 1 → tier 2 → stub, you'd fan out to N models and aggregate.
"""

from __future__ import annotations

import structlog
from prometheus_client import Counter, Histogram

from app.core.circuit_breaker import (
    CB_FALLBACKS,
    _ollama_primary_cb,
    _ollama_secondary_cb,
)
from app.core.config import settings

logger = structlog.get_logger(__name__)

ROUTER_LATENCY = Histogram(
    "model_router_latency_seconds",
    "End-to-end router latency",
    ["tier"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 15.0, 30.0],
)
ROUTER_TIER = Counter("model_router_tier_total", "Which tier was used", ["tier"])

# Stub response per agent type — used only when ALL tiers are OPEN
_STUB_RESPONSES = {
    "socratic": (
        "I'm having a bit of trouble connecting right now. "
        "What do you think the answer might be based on what we've covered?"
    ),
    "explain": (
        "My reasoning engine is briefly offline. "
        "Could you rephrase your question and try again in a moment?"
    ),
    "quiz": '{"question": "Service temporarily unavailable", "choices": {}, "correct_answer": "A", "explanation": "", "difficulty_b": 0.0}',
    "intent": "socratic",
    "memory": "Session summary temporarily unavailable.",
    "default": "I'm momentarily unavailable. Please try again in a few seconds.",
}


async def _call_ollama(
    model: str, messages: list[dict], temperature: float, max_tokens: int
) -> str:
    """Chat call via ChatOllama — no circuit breaker logic here."""
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    from langchain_ollama import ChatOllama

    lc_messages = []
    for m in messages:
        role, content = m["role"], m["content"]
        if role == "system":
            lc_messages.append(SystemMessage(content=content))
        elif role == "assistant":
            lc_messages.append(AIMessage(content=content))
        else:
            lc_messages.append(HumanMessage(content=content))

    llm = ChatOllama(
        model=model,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=temperature,
        num_predict=max_tokens,
        timeout=float(settings.OLLAMA_TIMEOUT_SECONDS),
    )
    response = await llm.ainvoke(lc_messages)
    return response.content.strip()


async def routed_chat(
    messages: list[dict],
    *,
    agent_type: str = "default",
    temperature: float = 0.7,
    max_tokens: int = 512,
) -> str:
    """
    Route an LLM chat call through the cascade:
      Tier 1 → primary Ollama model (with CB)
      Tier 2 → secondary Ollama model (with CB)
      Tier 3 → stub response (no CB — always succeeds)

    Never raises — always returns a string.
    """
    import time

    # ── Tier 1: Primary (fine-tuned Phi-3.5) ─────────────────────────────────
    if await _ollama_primary_cb.is_available():
        t0 = time.perf_counter()
        try:
            result = await _ollama_primary_cb.call(
                _call_ollama, settings.OLLAMA_MODEL, messages, temperature, max_tokens
            )
            ROUTER_LATENCY.labels(tier="primary").observe(time.perf_counter() - t0)
            ROUTER_TIER.labels(tier="primary").inc()
            logger.debug("router_tier1_success")
            return result
        except Exception as exc:
            logger.warning("router_tier1_failed", error=str(exc), agent_type=agent_type)
            CB_FALLBACKS.labels(service="ollama_primary", tier="tier1").inc()

    # ── Tier 2: Secondary (fallback Ollama model) ─────────────────────────────
    fallback_model = getattr(settings, "OLLAMA_FALLBACK_MODEL", "llama3.1:8b")
    if await _ollama_secondary_cb.is_available():
        t0 = time.perf_counter()
        try:
            result = await _ollama_secondary_cb.call(
                _call_ollama, fallback_model, messages, temperature, max_tokens
            )
            ROUTER_LATENCY.labels(tier="secondary").observe(time.perf_counter() - t0)
            ROUTER_TIER.labels(tier="secondary").inc()
            logger.warning("router_tier2_used", fallback_model=fallback_model)
            CB_FALLBACKS.labels(service="ollama_secondary", tier="tier2").inc()
            return result
        except Exception as exc:
            logger.error("router_tier2_failed", error=str(exc), agent_type=agent_type)
            CB_FALLBACKS.labels(service="ollama_secondary", tier="tier2_failed").inc()

    # ── Tier 3: Stub (graceful degradation — never fails) ────────────────────
    stub = _STUB_RESPONSES.get(agent_type, _STUB_RESPONSES["default"])
    ROUTER_TIER.labels(tier="stub").inc()
    logger.error("router_stub_used", agent_type=agent_type, reason="all_circuit_breakers_open")
    return stub


async def routed_chat_with_shadow(
    messages: list,
    *,
    agent_type: str = "default",
    temperature: float = 0.7,
    max_tokens: int = 512,
    session_id: str = "",
) -> str:
    """
    Like routed_chat but also fires a shadow call for A/B comparison.
    Use this in agents when shadow testing is active.
    """
    live = await routed_chat(
        messages,
        agent_type=agent_type,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    from app.core.shadow_testing import shadow_call

    await shadow_call(
        messages,
        live,
        agent_type=agent_type,
        session_id=session_id,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return live
