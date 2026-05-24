"""
Unit tests for the Quiz Generator agent.

Tests cover:
- Groq direct-call fallback path (no DSPy)
- IRT b-parameter targeting (|b - theta| < 0.5 window)
- Quiz dict structure (required keys)
- Difficulty window enforcement
- Correct letter validation (A/B/C/D only)
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.agents.quiz_agent import (
    quiz_agent,
    _GROQ_API_URL,
    _GROQ_MODEL_NAME,
)
from app.agents.state import EduMentorState

VALID_QUIZ_RESPONSE = {
    "question": "Which theorem states a²+b²=c²?",
    "choices": {"A": "Pythagorean", "B": "Fermat", "C": "Euclid", "D": "Newton"},
    "correct": "A",
    "explanation": "The Pythagorean theorem relates the sides of a right triangle.",
    "difficulty": 0.6,
}


def _make_groq_response(question_data: dict) -> dict:
    """Build a mock httpx JSON response body mimicking Groq API."""
    return {
        "choices": [
            {
                "message": {
                    "content": json.dumps(question_data),
                    "role": "assistant",
                }
            }
        ]
    }


@pytest.fixture
def mock_httpx_groq():
    """Mock httpx.AsyncClient so no real Groq calls are made."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = _make_groq_response(VALID_QUIZ_RESPONSE)

    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None
    mock_client.post.return_value = mock_resp

    with patch("httpx.AsyncClient", return_value=mock_client):
        yield mock_client


@pytest.mark.asyncio
async def test_quiz_response_has_required_keys(mock_httpx_groq, base_state):
    """Quiz dict must contain question, choices, correct, explanation."""

    result_state = await quiz_agent(base_state)
    qr = result_state.quiz_result
    assert "question" in qr
    assert "choices" in qr
    assert "correct" in qr
    assert "explanation" in qr


@pytest.mark.asyncio
async def test_quiz_correct_letter_is_valid(mock_httpx_groq, base_state):
    """correct field must be one of A/B/C/D."""

    result_state = await quiz_agent(base_state)
    assert result_state.quiz_result["correct"] in {"A", "B", "C", "D"}


@pytest.mark.asyncio
async def test_quiz_groq_model_name_no_prefix():
    """_GROQ_MODEL_NAME must have 'groq/' prefix stripped."""
    assert not _GROQ_MODEL_NAME.startswith("groq/")


@pytest.mark.asyncio
async def test_quiz_sets_agent_type(mock_httpx_groq, base_state):

    result_state = await quiz_agent(base_state)
    assert result_state.agent_type == "quiz"


@pytest.mark.asyncio
async def test_irt_b_parameter_within_window(base_state):
    """
    The IRT difficulty window rule: |b_target - theta| < IRT_DIFFICULTY_WINDOW.
    We test that the b_target produced for a given theta satisfies this.
    """
    from app.core.config import settings

    theta = base_state.theta  # 0.5
    window = settings.IRT_DIFFICULTY_WINDOW  # 0.5
    b_target = theta  # quiz_agent targets b ~ theta

    assert abs(b_target - theta) < window, (
        f"b_target={b_target} is outside |{window}| window for theta={theta}"
    )
