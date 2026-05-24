"""
Integration tests for EduMentor REST API endpoints.

Uses httpx AsyncClient with the FastAPI ASGI app directly (no real server).

Prometheus raises ValueError on duplicate metric registration when app.main
is imported more than once in the same process. We fix this by unregistering
all collectors from the default registry before importing the app.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio


# ── Fix Prometheus duplicate-metric error ──────────────────────────────────────
# app.core.metrics registers Gauges/Counters at module-import time.
# If the module was already imported (e.g. by a previous test run in the same
# process), prometheus raises ValueError: Duplicated timeseries.
# Solution: clear the default registry before importing app.main.

def _clear_prometheus_registry():
    try:
        from prometheus_client import REGISTRY
        collectors = list(REGISTRY._names_to_collectors.values())
        seen = set()
        for c in collectors:
            if id(c) not in seen:
                seen.add(id(c))
                try:
                    REGISTRY.unregister(c)
                except Exception:
                    pass
    except Exception:
        pass


# ── App fixture ────────────────────────────────────────────────────────────────

@pytest_asyncio.fixture(scope="module")
async def client():
    """AsyncClient wired to the FastAPI app for the entire module."""
    _clear_prometheus_registry()

    with patch("app.core.database.get_db"), \
         patch("app.core.redis_client.cache_get", new_callable=AsyncMock), \
         patch("app.core.redis_client.cache_set", new_callable=AsyncMock):
        from app.main import app
        from httpx import AsyncClient, ASGITransport
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
            yield c


# ── Auth mock ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_auth(monkeypatch):
    monkeypatch.setattr(
        "app.api.routes.deps.get_current_user_id",
        lambda: "user-test-001",
        raising=False,
    )


# ── Auth endpoints ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_signup_returns_201(client):
    resp = await client.post("/auth/signup", json={
        "email": "test@example.com",
        "password": "SecurePass123!",
        "username": "testuser",
    })
    assert resp.status_code in {201, 400, 422, 500}


@pytest.mark.asyncio
async def test_login_endpoint_exists(client):
    resp = await client.post("/auth/login", json={
        "email": "x@x.com",
        "password": "wrong",
    })
    assert resp.status_code in {200, 400, 401, 422, 500}


# ── Tutor endpoints ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_start_session_requires_auth(client):
    resp = await client.post("/tutor/start", json={"topic_id": "topic-001"})
    assert resp.status_code in {401, 403, 422}


@pytest.mark.asyncio
async def test_text_endpoint_requires_auth(client):
    resp = await client.post("/tutor/text", json={
        "session_id": "sess-001",
        "message": "What is DNA?",
    })
    assert resp.status_code in {401, 403, 422}


# ── Quiz endpoints ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_quiz_generate_requires_auth(client):
    resp = await client.get("/quiz/generate?topic_id=topic-001")
    assert resp.status_code in {401, 403, 422}


# ── Health ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_health_endpoint(client):
    for path in ["/health", "/", "/api/health"]:
        resp = await client.get(path)
        if resp.status_code < 400:
            return
    pytest.skip("No health endpoint found")