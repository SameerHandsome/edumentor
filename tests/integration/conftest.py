"""
Integration-level fixtures.

All external calls (Ollama, Groq, Redis, Qdrant, PostgreSQL) are replaced
with AsyncMocks so the LangGraph agent graph can be exercised end-to-end
without real infrastructure.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import AsyncClient

# ── FastAPI app ───────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def app():
    """
    Import the FastAPI app after env overrides are in place (set by the root
    conftest autouse fixture).  Module-scoped so the app is created once.
    """
    from app.main import app as _app
    return _app


@pytest_asyncio.fixture(scope="module")
async def client(app):
    from httpx import ASGITransport
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


# ── Service mocks ─────────────────────────────────────────────────────────────

@pytest.fixture
def mock_ollama_chat():
    with patch("app.agents.ollama_client.ollama_chat") as m:
        m.return_value = "This is a mocked agent response."
        yield m


@pytest.fixture
def mock_redis():
    with patch("app.core.redis_client.get_redis_pool") as m:
        pool_mock = AsyncMock()
        pool_mock.get.return_value = None
        pool_mock.set.return_value = True
        pool_mock.setex.return_value = True
        m.return_value = pool_mock
        yield pool_mock


@pytest.fixture
def mock_qdrant():
    with patch("app.rag.retriever.get_qdrant_client") as m:
        client_mock = MagicMock()
        client_mock.search.return_value = []
        m.return_value = client_mock
        yield client_mock


@pytest.fixture
def mock_db():
    with patch("app.core.database.get_db") as m:
        db_mock = AsyncMock()
        db_mock.execute.return_value = MagicMock(scalar_one_or_none=lambda: None)
        db_mock.commit = AsyncMock()
        m.return_value = db_mock
        yield db_mock
