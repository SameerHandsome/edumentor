"""
Shared pytest fixtures for the EduMentor test suite.
All external services (DB, Redis, Qdrant, Groq) are mocked here so tests
run in CI without real credentials.

IMPORTANT: environment variables are injected at module-import time before
any app module is loaded, because pydantic-settings reads them on first import.
"""
from __future__ import annotations

import asyncio
import os

# ── Inject env vars BEFORE any app import ─────────────────────────────────────
# pydantic-settings resolves Field(...) at import time, so these must be set
# before `from app...` appears anywhere in this file.
_TEST_ENV = {
    "DATABASE_URL": "postgresql+asyncpg://test:test@localhost/testdb",
    "SYNC_DATABASE_URL": "postgresql+psycopg2://test:test@localhost/testdb",
    "REDIS_URL": "rediss://test:test@localhost:6380",
    "REDIS_TOKEN": "fake-token",
    "QDRANT_URL": "http://localhost:6333",
    "QDRANT_API_KEY": "fake-key",
    "GROQ_API_KEY": "gsk_fake_groq_key_for_testing",
    "JWT_SECRET": "test-secret-key-32-chars-minimum!!",
    "CELERY_BROKER_URL": "memory://",
}
for _k, _v in _TEST_ENV.items():
    os.environ.setdefault(_k, _v)

# ── Now safe to import app modules ────────────────────────────────────────────
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agents.state import EduMentorState
from app.core.config import Settings


# ── Event loop ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def event_loop():
    """Single event loop for the entire test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

# ── Settings override ─────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def override_settings(monkeypatch):
    """
    Refresh settings cache each test so any per-test env changes take effect.
    """
    for k, v in _TEST_ENV.items():
        monkeypatch.setenv(k, v)

    from app.core import config as cfg_module
    cfg_module.get_settings.cache_clear()
    yield
    cfg_module.get_settings.cache_clear()


# ── Base EduMentorState ───────────────────────────────────────────────────────

@pytest.fixture
def base_state() -> EduMentorState:
    return EduMentorState(
        session_id="sess-test-001",
        user_id="user-test-001",
        topic_id="topic-test-001",
        topic_name="Mathematics",
        theta=0.5,
        student_level="intermediate",
        explanation_style="step_by_step",
        language="en",
        user_query="What is the Pythagorean theorem?",
        rag_chunks=["The Pythagorean theorem states a²+b²=c² for right triangles."],
        history=[],
    )


@pytest.fixture
def advanced_state(base_state) -> EduMentorState:
    return base_state.model_copy(update={
        "theta": 1.5,
        "student_level": "advanced",
        "user_query": "Derive the eigenvalue decomposition of a symmetric matrix.",
    })


@pytest.fixture
def beginner_state(base_state) -> EduMentorState:
    return base_state.model_copy(update={
        "theta": -1.0,
        "student_level": "beginner",
        "user_query": "What is addition?",
    })
