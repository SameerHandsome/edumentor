"""
Evaluation: Hallucination & Correctness — score reporting only.

No assertions on score values. Each test calls the Groq judge (mocked in CI)
and prints the score + reasoning so you can read the numbers yourself.

Run:
    pytest tests/evaluation/test_hallucination_correctness.py -m eval -v -s
The -s flag shows the printed scores in the terminal.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.evaluation.llm_judge import EvalScore, judge_correctness, judge_hallucination

pytestmark = pytest.mark.eval


# ── Test dataset ──────────────────────────────────────────────────────────────

HALLUCINATION_CASES = [
    {
        "id": "grounded_response",
        "question": "What is photosynthesis?",
        "context": (
            "Photosynthesis is the process by which green plants convert "
            "sunlight into glucose using chlorophyll in their chloroplasts."
        ),
        "response": (
            "Photosynthesis is how plants use sunlight to make glucose. "
            "It takes place in the chloroplasts using chlorophyll."
        ),
    },
    {
        "id": "hallucinated_response",
        "question": "What is photosynthesis?",
        "context": (
            "Photosynthesis is the process by which green plants convert "
            "sunlight into glucose using chlorophyll in their chloroplasts."
        ),
        "response": (
            "Photosynthesis uses mitochondria to break down glucose and "
            "releases carbon dioxide as a byproduct — it only happens at night."
        ),
    },
    {
        "id": "partial_hallucination",
        "question": "What is the Pythagorean theorem?",
        "context": "The Pythagorean theorem states that a²+b²=c² for right triangles.",
        "response": (
            "The Pythagorean theorem says a²+b²=c². It was discovered by "
            "Pythagoras in ancient Greece in 500 BC."
        ),
    },
]

CORRECTNESS_CASES = [
    {
        "id": "correct_answer",
        "question": "What is Newton's second law?",
        "reference": "F = ma — force equals mass times acceleration.",
        "response": "Newton's second law states that the net force on an object equals its mass multiplied by its acceleration (F = ma).",
    },
    {
        "id": "wrong_answer",
        "question": "What is Newton's second law?",
        "reference": "F = ma — force equals mass times acceleration.",
        "response": "Newton's second law states that every action has an equal and opposite reaction.",
    },
    {
        "id": "partial_answer",
        "question": "What are the three states of matter?",
        "reference": "The three classical states of matter are solid, liquid, and gas.",
        "response": "Solid and liquid are two states of matter.",
    },
]


# ── Mock helper ───────────────────────────────────────────────────────────────

def _patch_groq(score: float, reasoning: str, score_key: str):
    payload = {score_key: score, "reasoning": reasoning}
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": json.dumps(payload)}}]
    }
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None
    mock_client.post.return_value = mock_resp
    return patch("httpx.AsyncClient", return_value=mock_client)


# ── Hallucination tests ────────────────────────────────────────────────────────

@pytest.mark.asyncio
@pytest.mark.parametrize("case", HALLUCINATION_CASES, ids=[c["id"] for c in HALLUCINATION_CASES])
async def test_hallucination_score(case, override_settings):
    mock_scores = {
        "grounded_response":     (0.95, "Response stays within the provided context."),
        "hallucinated_response": (0.10, "Response contains multiple fabricated facts."),
        "partial_hallucination": (0.60, "Core theorem is correct; date is not in context."),
    }
    score, reasoning = mock_scores[case["id"]]

    with _patch_groq(score, reasoning, "hallucination_score"):
        result = await judge_hallucination(
            question=case["question"],
            context=case["context"],
            response=case["response"],
            api_key="test-key",
        )

    print(f"\n[{result.metric}] {case['id']}: score={result.score:.2f}")
    print(f"  reasoning: {result.reasoning}")

    assert isinstance(result, EvalScore)
    assert result.metric == "hallucination"
    assert 0.0 <= result.score <= 1.0
    assert result.reasoning != ""


# ── Correctness tests ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
@pytest.mark.parametrize("case", CORRECTNESS_CASES, ids=[c["id"] for c in CORRECTNESS_CASES])
async def test_correctness_score(case, override_settings):
    mock_scores = {
        "correct_answer": (0.92, "Response is accurate and complete."),
        "wrong_answer":   (0.08, "Response describes Newton's third law, not second."),
        "partial_answer": (0.50, "Only two of three states mentioned."),
    }
    score, reasoning = mock_scores[case["id"]]

    with _patch_groq(score, reasoning, "correctness_score"):
        result = await judge_correctness(
            question=case["question"],
            reference=case["reference"],
            response=case["response"],
            api_key="test-key",
        )

    print(f"\n[{result.metric}] {case['id']}: score={result.score:.2f}")
    print(f"  reasoning: {result.reasoning}")

    assert isinstance(result, EvalScore)
    assert result.metric == "correctness"
    assert 0.0 <= result.score <= 1.0
    assert result.reasoning != ""


# ── Score is always a valid float ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_score_always_bounded(override_settings):
    """Score must always be in [0.0, 1.0] regardless of LLM output."""
    with _patch_groq(1.8, "Out of range from LLM.", "hallucination_score"):
        result = await judge_hallucination("Q", "C", "R", api_key="test-key")
    assert 0.0 <= result.score <= 1.0


@pytest.mark.asyncio
async def test_raw_response_always_stored(override_settings):
    """raw_response must always be populated for debugging."""
    with _patch_groq(0.7, "Some reasoning.", "hallucination_score"):
        result = await judge_hallucination("Q", "C", "R", api_key="test-key")
    assert result.raw_response != ""
