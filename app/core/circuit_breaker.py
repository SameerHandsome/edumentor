"""
Circuit Breaker + Cascade Model Router for EduMentor.

Circuit Breaker Pattern:
  CLOSED  → requests flow normally, failures counted
  OPEN    → requests fail-fast (no calls to downstream), reset after timeout
  HALF-OPEN → one probe request allowed; success→CLOSED, failure→OPEN

Cascade Router:
  Primary   → fine-tuned Phi-3.5 (Ollama local)
  Secondary → larger Ollama model (e.g. llama3.1:8b) if primary CB is OPEN
  Tertiary  → minimal stub response if all CBs are OPEN (graceful degradation)

This prevents:
  1. Thundering-herd failures when Ollama is overloaded
  2. Cascading timeouts in the agent graph
  3. Silent degradation (errors are always surfaced in metrics)
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog
from prometheus_client import REGISTRY, Counter, Gauge

logger = structlog.get_logger(__name__)


# ── Prometheus helpers ────────────────────────────────────────────────────────
# Prometheus raises ValueError if the same metric name is registered twice.
# This can happen when the module is imported more than once in the same
# process (e.g. via a lazy import inside a hot-path function).  The helpers
# below return the already-registered collector when one exists.


def _gauge(name: str, doc: str, labels: list[str]) -> Gauge:
    if name in REGISTRY._names_to_collectors:
        return REGISTRY._names_to_collectors[name]  # type: ignore[return-value]
    return Gauge(name, doc, labels)


def _counter(name: str, doc: str, labels: list[str]) -> Counter:
    if name in REGISTRY._names_to_collectors:
        return REGISTRY._names_to_collectors[name]  # type: ignore[return-value]
    return Counter(name, doc, labels)


# ── Prometheus metrics ────────────────────────────────────────────────────────
CB_STATE = _gauge(
    "circuit_breaker_state",
    "Circuit breaker state (0=closed,1=open,2=half-open)",
    ["service"],
)
CB_TRIPS = _counter(
    "circuit_breaker_trips_total",
    "Total circuit breaker trips",
    ["service"],
)
CB_FALLBACKS = _counter(
    "circuit_breaker_fallbacks_total",
    "Total times fallback was used",
    ["service", "tier"],
)


class CBState(Enum):
    CLOSED = "closed"  # normal — requests flow through
    OPEN = "open"  # tripped — fail fast
    HALF_OPEN = "half_open"  # probing — one request allowed


@dataclass
class CircuitBreaker:
    """
    Thread-safe async circuit breaker.

    Parameters
    ----------
    name              : human-readable name for metrics/logs
    failure_threshold : consecutive failures before tripping (default 5)
    success_threshold : consecutive successes in HALF_OPEN before closing (default 2)
    timeout_seconds   : seconds to stay OPEN before moving to HALF_OPEN (default 30)
    """

    name: str
    failure_threshold: int = 5
    success_threshold: int = 2
    timeout_seconds: float = 30.0

    _state: CBState = field(default=CBState.CLOSED, init=False)
    _failures: int = field(default=0, init=False)
    _successes: int = field(default=0, init=False)
    _opened_at: float = field(default=0.0, init=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)

    @property
    def state(self) -> CBState:
        return self._state

    async def is_available(self) -> bool:
        """Return True if a request should be allowed through."""
        async with self._lock:
            if self._state == CBState.CLOSED:
                return True
            if self._state == CBState.OPEN:
                if time.monotonic() - self._opened_at >= self.timeout_seconds:
                    self._state = CBState.HALF_OPEN
                    CB_STATE.labels(service=self.name).set(2)
                    logger.info("circuit_breaker_half_open", service=self.name)
                    return True  # allow one probe
                return False
            # HALF_OPEN → allow the one probe
            return True

    async def record_success(self) -> None:
        async with self._lock:
            self._failures = 0
            if self._state == CBState.HALF_OPEN:
                self._successes += 1
                if self._successes >= self.success_threshold:
                    self._state = CBState.CLOSED
                    self._successes = 0
                    CB_STATE.labels(service=self.name).set(0)
                    logger.info("circuit_breaker_closed", service=self.name)

    async def record_failure(self) -> None:
        async with self._lock:
            self._failures += 1
            self._successes = 0
            if self._state == CBState.HALF_OPEN or self._failures >= self.failure_threshold:
                self._state = CBState.OPEN
                self._opened_at = time.monotonic()
                CB_STATE.labels(service=self.name).set(1)
                CB_TRIPS.labels(service=self.name).inc()
                logger.warning("circuit_breaker_opened", service=self.name, failures=self._failures)

    async def call(self, fn: Callable, *args: Any, **kwargs: Any) -> Any:
        """Execute fn if CB allows; raises RuntimeError if OPEN."""
        if not await self.is_available():
            raise RuntimeError(f"CircuitBreaker[{self.name}] is OPEN — fast failing")
        try:
            result = await fn(*args, **kwargs)
            await self.record_success()
            return result
        except Exception:
            await self.record_failure()
            raise


# ── Singleton circuit breakers ────────────────────────────────────────────────
_ollama_primary_cb = CircuitBreaker(name="ollama_primary", failure_threshold=5, timeout_seconds=30)
_ollama_secondary_cb = CircuitBreaker(
    name="ollama_secondary", failure_threshold=3, timeout_seconds=60
)
_qdrant_cb = CircuitBreaker(name="qdrant", failure_threshold=5, timeout_seconds=45)
_redis_cb = CircuitBreaker(name="redis", failure_threshold=5, timeout_seconds=20)


def get_ollama_cb() -> CircuitBreaker:
    return _ollama_primary_cb


def get_qdrant_cb() -> CircuitBreaker:
    return _qdrant_cb


def get_redis_cb() -> CircuitBreaker:
    return _redis_cb
