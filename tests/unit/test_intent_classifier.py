"""
Unit tests for the Intent Classifier agent.

Tests that:
- Each valid intent is returned unchanged
- Unknown intent falls back to 'socratic'
- State is updated correctly
- ollama_chat is called with correct temperature / max_tokens
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.agents.intent_classifier import classify_intent, VALID_INTENTS
from app.agents.state import EduMentorState


@pytest.fixture
def mock_ollama():
    with patch("app.agents.intent_classifier.ollama_chat") as m:
        yield m


@pytest.mark.asyncio
@pytest.mark.parametrize("raw_response,expected_intent", [
    ("socratic", "socratic"),
    ("quiz", "quiz"),
    ("explain", "explain"),
    ("feedback", "feedback"),
    ("meta", "meta"),
    ("QUIZ\n", "quiz"),          # strips + lowercases
    ("  explain  ", "explain"),  # strips whitespace
])
async def test_valid_intents_returned(mock_ollama, base_state, raw_response, expected_intent):
    """Classifier correctly maps raw LLM output to valid intents."""
    mock_ollama.return_value = raw_response
    result = await classify_intent(base_state)
    assert result.intent == expected_intent


@pytest.mark.asyncio
@pytest.mark.parametrize("garbage", [
    "I think this is a quiz question",
    "unknown",
    "none",
    "123",
    "",
])
async def test_unknown_intent_falls_back_to_socratic(mock_ollama, base_state, garbage):
    """Any response not in VALID_INTENTS should default to 'socratic'."""
    mock_ollama.return_value = garbage
    result = await classify_intent(base_state)
    assert result.intent == "socratic"


@pytest.mark.asyncio
async def test_intent_low_temperature_call(mock_ollama, base_state):
    """Intent classifier must use temperature=0.1 and max_tokens=10 (cheap call)."""
    mock_ollama.return_value = "quiz"
    await classify_intent(base_state)
    mock_ollama.assert_called_once()
    _, kwargs = mock_ollama.call_args
    assert kwargs.get("temperature") == 0.1
    assert kwargs.get("max_tokens") == 10


@pytest.mark.asyncio
async def test_intent_state_unchanged_except_intent(mock_ollama, base_state):
    """classify_intent must not mutate any field other than 'intent'."""
    mock_ollama.return_value = "explain"
    original_query = base_state.user_query
    result = await classify_intent(base_state)
    assert result.user_query == original_query
    assert result.session_id == base_state.session_id
    assert result.theta == base_state.theta


@pytest.mark.asyncio
async def test_valid_intents_constant():
    """VALID_INTENTS set must include all five expected intents."""
    assert VALID_INTENTS == {"socratic", "quiz", "explain", "feedback", "meta"}
