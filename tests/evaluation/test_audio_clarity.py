"""
Evaluation: Audio / TTS Clarity — score reporting only.

EduMentor uses Coqui TTS for voice responses. These tests evaluate whether
agent responses are appropriate for text-to-speech playback.

No assertions on score values — just print scores so you can judge quality.

Run:
    pytest tests/evaluation/test_audio_clarity.py -m eval -v -s
"""
from __future__ import annotations

import json
import re
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.evaluation.llm_judge import EvalScore, judge_audio_clarity

pytestmark = pytest.mark.eval


# ── Heuristic helpers (pre-judge fast checks, no LLM needed) ─────────────────

_MARKDOWN_RE = re.compile(r"(\*{1,3}|_{1,3}|#{1,6}|~~|`)")
_LATEX_RE    = re.compile(r"(\\[a-zA-Z]+\{|\\frac|\\int|\\sum|\$)")


def has_markdown(text: str) -> bool:
    return bool(_MARKDOWN_RE.search(text))


def has_latex(text: str) -> bool:
    return bool(_LATEX_RE.search(text))


def sentence_count(text: str) -> int:
    parts = re.split(r"[.!?]+", text.strip())
    return len([p for p in parts if p.strip()])


def max_sentence_words(text: str) -> int:
    sentences = re.split(r"[.!?]+", text.strip())
    return max((len(s.split()) for s in sentences if s.strip()), default=0)


# ── Test dataset ──────────────────────────────────────────────────────────────

AUDIO_CASES = [
    {
        "id": "clean_conversational",
        "response": (
            "Imagine a triangle sitting on a flat surface. "
            "Now, what happens if you square each side length? "
            "Try adding the two shorter squared values together."
        ),
    },
    {
        "id": "markdown_heavy",
        "response": (
            "**Photosynthesis** converts light into glucose.\n"
            "## Key equation:\n`6CO₂ + 6H₂O → C₆H₁₂O₆ + 6O₂`\n"
            "- Requires chlorophyll\n- Produces oxygen"
        ),
    },
    {
        "id": "latex_heavy",
        "response": (
            r"The Pythagorean theorem is expressed as $a^2 + b^2 = c^2$. "
            r"For eigenvalues: \( \lambda \mathbf{v} = A\mathbf{v} \). "
            r"Integration: \int_{0}^{\infty} e^{-x} dx = 1."
        ),
    },
    {
        "id": "too_many_sentences",
        "response": (
            "So photosynthesis is really interesting. "
            "First you need to understand chlorophyll. "
            "Then you need to think about ATP production. "
            "Don't forget about the light reactions. "
            "And also the Calvin cycle is very important here."
        ),
    },
    {
        "id": "long_sentences",
        "response": (
            "The process of photosynthesis is a very complex biochemical reaction "
            "that occurs inside the chloroplasts of plant cells and involves the "
            "absorption of sunlight by chlorophyll molecules to drive the conversion "
            "of carbon dioxide and water into glucose and oxygen."
        ),
    },
    {
        "id": "mixed_symbols_and_prose",
        "response": (
            "The formula F = ma is Newton's second law. "
            "This means force equals mass times acceleration. "
            "So if you double the mass, the force doubles too."
        ),
    },
]


# ── Mock helper ───────────────────────────────────────────────────────────────

def _patch_groq(score: float, reasoning: str, issues: list | None = None):
    payload = {
        "audio_clarity_score": score,
        "reasoning": reasoning,
        "tts_issues": issues or [],
    }
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


# ── Heuristic unit tests (no LLM, no mock needed) ─────────────────────────────

def test_heuristic_markdown_detected():
    assert has_markdown("**Bold** and *italic*")
    assert not has_markdown("Plain text response.")


def test_heuristic_latex_detected():
    assert has_latex(r"The formula is $a^2 + b^2 = c^2$")
    assert not has_latex("The formula is a squared plus b squared equals c squared.")


def test_heuristic_sentence_count():
    text = "First sentence. Second sentence! Third one?"
    assert sentence_count(text) == 3


def test_heuristic_max_sentence_words():
    text = "Short. This is a much longer sentence with many many many many words in it."
    assert max_sentence_words(text) > 5


def test_clean_response_passes_heuristics():
    case = next(c for c in AUDIO_CASES if c["id"] == "clean_conversational")
    assert not has_markdown(case["response"])
    assert not has_latex(case["response"])
    assert sentence_count(case["response"]) <= 4


def test_markdown_response_detected_by_heuristic():
    case = next(c for c in AUDIO_CASES if c["id"] == "markdown_heavy")
    assert has_markdown(case["response"])


def test_latex_response_detected_by_heuristic():
    case = next(c for c in AUDIO_CASES if c["id"] == "latex_heavy")
    assert has_latex(case["response"])


# ── LLM judge tests — score reporting only ────────────────────────────────────

_MOCK_SCORES = {
    "clean_conversational": (
        0.95,
        "Short sentences, no markdown, natural speech flow.",
        [],
    ),
    "markdown_heavy": (
        0.08,
        "Contains headers, bold, code block and bullet lists — unreadable by TTS.",
        ["markdown headers", "bold text", "code blocks", "bullet lists"],
    ),
    "latex_heavy": (
        0.10,
        "LaTeX math expressions and inline math symbols cannot be spoken aloud.",
        ["LaTeX math expressions", "inline math $...$ notation"],
    ),
    "too_many_sentences": (
        0.55,
        "Five sentences — slightly over the four-sentence voice limit.",
        ["5 sentences (limit is 4)"],
    ),
    "long_sentences": (
        0.40,
        "Single sentence exceeds 40 words — too long for comfortable TTS playback.",
        ["sentence exceeds 20-word guideline"],
    ),
    "mixed_symbols_and_prose": (
        0.75,
        "Mostly clean; inline 'F = ma' is borderline but readable.",
        [],
    ),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("case", AUDIO_CASES, ids=[c["id"] for c in AUDIO_CASES])
async def test_audio_clarity_score(case, override_settings):
    score, reasoning, issues = _MOCK_SCORES[case["id"]]

    with _patch_groq(score, reasoning, issues):
        result = await judge_audio_clarity(
            response=case["response"],
            api_key="test-key",
        )

    # ── Print score for human inspection ──────────────────────────────────────
    print(f"\n[{result.metric}] {case['id']}: score={result.score:.2f}")
    print(f"  reasoning: {result.reasoning}")

    # ── Only structural assertions — no score thresholds ──────────────────────
    assert isinstance(result, EvalScore)
    assert result.metric == "audio_clarity"
    assert 0.0 <= result.score <= 1.0
    assert result.reasoning != ""
    assert result.raw_response != ""


@pytest.mark.asyncio
async def test_score_clamped_when_llm_returns_out_of_range(override_settings):
    """If Groq returns a score > 1.0, it must be clamped to 1.0."""
    with _patch_groq(1.5, "Out of range."):
        result = await judge_audio_clarity("Some text.", api_key="test-key")
    assert 0.0 <= result.score <= 1.0
