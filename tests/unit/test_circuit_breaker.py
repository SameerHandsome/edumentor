"""
Unit tests for the Circuit Breaker (app/core/circuit_breaker.py).

Tests state transitions:
  CLOSED → OPEN (on failure threshold)
  OPEN → HALF_OPEN (after timeout)
  HALF_OPEN → CLOSED (on success threshold)
  HALF_OPEN → OPEN (on failure)
"""
from __future__ import annotations

import time
from unittest.mock import patch, AsyncMock

import pytest

from app.core.circuit_breaker import CircuitBreaker, CBState


@pytest.fixture
def cb():
    """Fresh circuit breaker with low thresholds for fast testing."""
    return CircuitBreaker(
        name="test_cb",
        failure_threshold=3,
        success_threshold=1,   # 1 success to re-close in HALF_OPEN
        timeout_seconds=1,     # 1 second — fast tests
    )


@pytest.mark.asyncio
async def test_initial_state_is_closed(cb):
    assert cb.state == CBState.CLOSED


@pytest.mark.asyncio
async def test_opens_after_failure_threshold(cb):
    for _ in range(3):
        await cb.record_failure()
    assert cb.state == CBState.OPEN


@pytest.mark.asyncio
async def test_does_not_open_below_threshold(cb):
    for _ in range(2):
        await cb.record_failure()
    assert cb.state == CBState.CLOSED


@pytest.mark.asyncio
async def test_success_resets_failure_count(cb):
    await cb.record_failure()
    await cb.record_failure()
    await cb.record_success()
    # failure count reset — need 3 more to open
    assert cb.state == CBState.CLOSED
    for _ in range(2):
        await cb.record_failure()
    assert cb.state == CBState.CLOSED


@pytest.mark.asyncio
async def test_transitions_to_half_open_after_timeout(cb):
    for _ in range(3):
        await cb.record_failure()
    assert cb.state == CBState.OPEN

    # Simulate timeout by patching time.monotonic
    with patch("time.monotonic", return_value=time.monotonic() + 5):
        allowed = await cb.is_available()
        assert allowed is True
        assert cb.state == CBState.HALF_OPEN


@pytest.mark.asyncio
async def test_half_open_success_closes_breaker(cb):
    for _ in range(3):
        await cb.record_failure()

    with patch("time.monotonic", return_value=time.monotonic() + 5):
        await cb.is_available()  # transitions to HALF_OPEN
        await cb.record_success()
    assert cb.state == CBState.CLOSED


@pytest.mark.asyncio
async def test_half_open_failure_reopens_breaker(cb):
    for _ in range(3):
        await cb.record_failure()

    with patch("time.monotonic", return_value=time.monotonic() + 5):
        await cb.is_available()  # transitions to HALF_OPEN
        await cb.record_failure()
    assert cb.state == CBState.OPEN


@pytest.mark.asyncio
async def test_open_rejects_requests(cb):
    for _ in range(3):
        await cb.record_failure()
    # No timeout — still OPEN
    assert await cb.is_available() is False


@pytest.mark.asyncio
async def test_call_raises_when_open(cb):
    for _ in range(3):
        await cb.record_failure()

    async def dummy():
        return "ok"

    with pytest.raises(RuntimeError, match="OPEN"):
        await cb.call(dummy)
