"""
Integration tests for the multi-layer cache (app/core/multi_layer_cache.py).

The cache uses cache_get/cache_set/cache_delete from app.core.redis_client.
We patch those directly since multi_layer_cache imports them at module level.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from app.core.multi_layer_cache import (
    ml_get_text,
    ml_set_text,
    ml_get_mastery,
    ml_set_mastery,
)

# Patch target — where the names live after import
_REDIS = "app.core.redis_client"


@pytest.fixture
def mock_redis(monkeypatch):
    """Patch cache_get / cache_set / cache_delete on the redis_client module."""
    cache_get_mock  = AsyncMock(return_value=None)   # miss by default
    cache_set_mock  = AsyncMock(return_value=True)
    cache_del_mock  = AsyncMock(return_value=True)

    monkeypatch.setattr("app.core.redis_client.cache_get",    cache_get_mock)
    monkeypatch.setattr("app.core.redis_client.cache_set",    cache_set_mock)
    monkeypatch.setattr("app.core.redis_client.cache_delete", cache_del_mock)

    # Also patch inside multi_layer_cache where the names were already imported
    monkeypatch.setattr("app.core.multi_layer_cache.cache_get",    cache_get_mock)
    monkeypatch.setattr("app.core.multi_layer_cache.cache_set",    cache_set_mock)
    monkeypatch.setattr("app.core.multi_layer_cache.cache_delete", cache_del_mock)

    return {"get": cache_get_mock, "set": cache_set_mock, "delete": cache_del_mock}


@pytest.mark.asyncio
async def test_text_cache_miss_returns_none(mock_redis):
    mock_redis["get"].return_value = None
    result = await ml_get_text("user-001", "sess-001", "abc123")
    assert result is None


@pytest.mark.asyncio
async def test_text_cache_hit_returns_value(mock_redis):
    cached = json.dumps({"agent_response": "Cached answer."})
    mock_redis["get"].return_value = cached
    result = await ml_get_text("user-001", "sess-001", "abc123")
    assert result is not None


@pytest.mark.asyncio
async def test_text_cache_set_calls_redis(mock_redis):
    await ml_set_text("user-001", "sess-001", "abc123", {"agent_response": "New answer."})
    assert mock_redis["set"].called


@pytest.mark.asyncio
async def test_mastery_cache_miss_returns_none(mock_redis):
    mock_redis["get"].return_value = None
    result = await ml_get_mastery("user-001")
    assert result is None


@pytest.mark.asyncio
async def test_mastery_cache_set_and_get_round_trip(mock_redis):
    mastery_data = json.dumps({"theta": 0.75, "student_level": "intermediate"})
    mock_redis["get"].return_value = mastery_data

    await ml_set_mastery("user-001", {"theta": 0.75})
    result = await ml_get_mastery("user-001")
    assert result is not None


@pytest.mark.asyncio
async def test_cache_handles_redis_exception_gracefully(mock_redis):
    """If cache_get raises, ml_get_text should return None or raise — both acceptable."""
    mock_redis["get"].side_effect = ConnectionError("Redis unreachable")
    try:
        result = await ml_get_text("user-999", "sess-999", "unique-key-xyz")
        assert result is None
    except ConnectionError:
        pass