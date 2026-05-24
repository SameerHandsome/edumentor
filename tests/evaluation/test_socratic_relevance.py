"""
Evaluation: Socratic Quality & Relevance — score reporting only.

No assertions on score values. Tests report scores so you can judge quality.

Run:
    pytest tests/evaluation/test_socratic_relevance.py -m eval -v -s
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.evaluation.llm_judge import EvalScore, judge_relevance, judge_socratic_quality

pytestmark = pytest.mark.eval


# ── Datasets ──────────────────────────────────────────────────────────────────

SOCRATIC_CASES = [
    {
        "id": "guides_with_question",
        "question": "What is the Pythagorean theorem?",
        "response": (
            "Interesting question! Think about a right triangle — "
            "what relationship do you notice between its sides? "
            "Can you draw one and label the longest side?"
        ),
    },
    {
        "id": "gives_direct_answer",
        "question": "What is the Pythagorean theorem?",
        "response": (
            "The Pythagorean theorem states that a²+b²=c² for right triangles, "
            "where c is the hypotenuse."
        ),
    },
    {
        "id": "uses_analogy_hint",
        "question": "How does photosynthesis work?",
        "response": (
            "Great question! You already know plants need sunlight — "
            "what do you think they might do with that energy?"
        ),
    },
    {
        "id": "vague_non_answer",
        "question": "Explain Newton's second law.",
        "response": "That's a really interesting topic. Physics is everywhere around us!",
    },
]

RELEVANCE_CASES = [
    {
        "id": "on_topic",
        "question": "What is Newton's first law?",
        "response": (
            "Think about a hockey puck sliding on ice — "
            "what happens to it if nothing pushes or pulls it?"
        ),
    },
    {
        "id": "off_topic",
        "question": "What is Newton's first law?",
        "response": "I'd love to tell you about my favourite recipe for pasta carbonara!",
    },
    {
        "id": "tangentially_related",
        "question": "What is the speed of light?",
        "response": "That's a great general physics question. Forces play a big role in motion.",
    },
    {
        "id": "partially_relevant",
        "question": "Explain the water cycle.",
        "response": "Water is very important for life. Clouds form in the sky.",
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


# ── Socratic quality tests ─────────────────────────────────────────────────────

@pytest.mark.asyncio
@pytest.mark.parametrize("case", SOCRATIC_CASES, ids=[c["id"] for c in SOCRATIC_CASES])
async def test_socratic_quality_score(case, override_settings):
    mock_scores = {
        "guides_with_question": (0.92, "Response asks guiding questions without revealing the answer."),
        "gives_direct_answer":  (0.05, "Response gives the answer directly — violates Socratic rule."),
        "uses_analogy_hint":    (0.88, "Uses a hint to prompt student thinking."),
        "vague_non_answer":     (0.40, "Avoids giving an answer but provides no useful guidance either."),
    }
    score, reasoning = mock_scores[case["id"]]

    with _patch_groq(score, reasoning, "socratic_quality_score"):
        result = await judge_socratic_quality(
            question=case["question"],
            response=case["response"],
            api_key="test-key",
        )

    print(f"\n[{result.metric}] {case['id']}: score={result.score:.2f}")
    print(f"  reasoning: {result.reasoning}")

    assert isinstance(result, EvalScore)
    assert result.metric == "socratic_quality"
    assert 0.0 <= result.score <= 1.0
    assert result.reasoning != ""


# ── Relevance tests ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
@pytest.mark.parametrize("case", RELEVANCE_CASES, ids=[c["id"] for c in RELEVANCE_CASES])
async def test_relevance_score(case, override_settings):
    mock_scores = {
        "on_topic":            (0.95, "Response is directly relevant to Newton's first law."),
        "off_topic":           (0.02, "Response is about pasta — completely unrelated."),
        "tangentially_related":(0.35, "Mentions physics but does not address speed of light."),
        "partially_relevant":  (0.50, "Mentions water but misses the cycle explanation."),
    }
    score, reasoning = mock_scores[case["id"]]

    with _patch_groq(score, reasoning, "relevance_score"):
        result = await judge_relevance(
            question=case["question"],
            response=case["response"],
            api_key="test-key",
        )

    print(f"\n[{result.metric}] {case['id']}: score={result.score:.2f}")
    print(f"  reasoning: {result.reasoning}")

    assert isinstance(result, EvalScore)
    assert result.metric == "relevance"
    assert 0.0 <= result.score <= 1.0
    assert result.reasoning != ""
