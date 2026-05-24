"""
End-to-end evaluation suite — full score report across all metrics.

Runs every metric on every case in EVAL_DATASET and prints a summary table.
No test fails based on score values — the table is your signal.

Run:
    pytest tests/evaluation/test_eval_suite.py -m eval -v -s

The -s flag shows the full printed table in the terminal output.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.evaluation.llm_judge import (
    EvalScore,
    judge_audio_clarity,
    judge_correctness,
    judge_hallucination,
    judge_relevance,
    judge_socratic_quality,
)

pytestmark = pytest.mark.eval


# ── Dataset ───────────────────────────────────────────────────────────────────

EVAL_DATASET = [
    {
        "id": "math_socratic_good",
        "agent": "socratic",
        "question": "What is the Pythagorean theorem?",
        "context": "The Pythagorean theorem: a²+b²=c² for right triangles.",
        "reference": "a²+b²=c²",
        "response": (
            "Great thinking so far! If you draw a right triangle and label its sides, "
            "what pattern do you notice when you square each side length?"
        ),
        "is_voice": True,
    },
    {
        "id": "math_socratic_bad",
        "agent": "socratic",
        "question": "What is the Pythagorean theorem?",
        "context": "The Pythagorean theorem: a²+b²=c² for right triangles.",
        "reference": "a²+b²=c²",
        "response": "The Pythagorean theorem states that a²+b²=c² where c is the hypotenuse.",
        "is_voice": False,
    },
    {
        "id": "physics_explainer_clean",
        "agent": "explainer",
        "question": "Explain Newton's first law.",
        "context": (
            "Newton's first law: an object at rest stays at rest, and an object "
            "in motion stays in motion, unless acted upon by an external force."
        ),
        "reference": "An object stays in its current state of motion unless a net external force acts on it.",
        "response": (
            "Think of a hockey puck gliding on frictionless ice. "
            "Nothing stops it — it keeps sliding forever. "
            "That is Newton's first law: objects resist changes to their motion."
        ),
        "is_voice": True,
    },
    {
        "id": "biology_hallucinated",
        "agent": "explainer",
        "question": "What is DNA?",
        "context": (
            "DNA (deoxyribonucleic acid) carries genetic information in living organisms. "
            "It is a double helix structure made of nucleotides."
        ),
        "reference": "DNA is a double-helix molecule that carries genetic information.",
        "response": (
            "DNA is a triple-helix molecule found in the nucleus that carries "
            "genetic information and generates electricity for the cell."
        ),
        "is_voice": False,
    },
    {
        "id": "chemistry_correct",
        "agent": "explainer",
        "question": "Which element has atomic number 6?",
        "context": "Carbon has atomic number 6. It is the basis of organic chemistry.",
        "reference": "Carbon",
        "response": (
            "The element with atomic number 6 is Carbon. "
            "It is the foundation of organic chemistry."
        ),
        "is_voice": True,
    },
    {
        "id": "physics_off_topic",
        "agent": "explainer",
        "question": "What is Newton's first law?",
        "context": "Newton's first law describes inertia.",
        "reference": "Objects in motion stay in motion unless acted upon by a force.",
        "response": "I love cooking! Have you ever tried making pasta from scratch?",
        "is_voice": False,
    },
    {
        "id": "math_markdown_response",
        "agent": "explainer",
        "question": "What is the quadratic formula?",
        "context": "The quadratic formula is x = (-b ± √(b²-4ac)) / 2a.",
        "reference": "x = (-b ± √(b²-4ac)) / 2a",
        "response": (
            "## Quadratic Formula\n"
            "The formula is:\n"
            "$$x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$$\n"
            "Where **a**, **b**, **c** are coefficients."
        ),
        "is_voice": True,  # intentionally voice=True to expose TTS problem
    },
]

# Pre-defined mock scores for each (case_id, metric) pair
# These simulate realistic Groq judge responses
_MOCK_SCORES: dict[tuple[str, str], tuple[float, str]] = {
    ("math_socratic_good",    "hallucination"):    (0.90, "Response grounded in question; no fabrication."),
    ("math_socratic_good",    "correctness"):      (0.80, "Doesn't reveal answer but guides correctly."),
    ("math_socratic_good",    "relevance"):        (0.95, "Directly relevant to the question."),
    ("math_socratic_good",    "socratic_quality"): (0.93, "Asks guiding question without giving the answer."),
    ("math_socratic_good",    "audio_clarity"):    (0.91, "Short sentences, no symbols, conversational."),

    ("math_socratic_bad",     "hallucination"):    (0.85, "Formula is correct and grounded."),
    ("math_socratic_bad",     "correctness"):      (0.95, "Correct answer given directly."),
    ("math_socratic_bad",     "relevance"):        (0.97, "Directly answers the question."),
    ("math_socratic_bad",     "socratic_quality"): (0.03, "Gives direct answer — violates Socratic rule."),
    # is_voice=False so no audio_clarity

    ("physics_explainer_clean", "hallucination"):  (0.92, "All claims supported by context."),
    ("physics_explainer_clean", "correctness"):    (0.90, "Correctly explains inertia via analogy."),
    ("physics_explainer_clean", "relevance"):      (0.96, "Directly relevant to Newton's first law."),
    ("physics_explainer_clean", "audio_clarity"):  (0.94, "Three short sentences, no symbols."),

    ("biology_hallucinated",  "hallucination"):    (0.05, "Triple helix and electricity generation are fabricated."),
    ("biology_hallucinated",  "correctness"):      (0.04, "Completely contradicts the reference."),
    ("biology_hallucinated",  "relevance"):        (0.70, "About DNA, but content is wrong."),
    # is_voice=False so no audio_clarity

    ("chemistry_correct",     "hallucination"):    (0.97, "Fully grounded in context."),
    ("chemistry_correct",     "correctness"):      (0.98, "Correct and complete."),
    ("chemistry_correct",     "relevance"):        (0.99, "Directly answers the question."),
    ("chemistry_correct",     "audio_clarity"):    (0.93, "Two clean sentences, spoken naturally."),

    ("physics_off_topic",     "hallucination"):    (0.50, "No facts to check — response is off-topic."),
    ("physics_off_topic",     "correctness"):      (0.01, "Does not address Newton's first law at all."),
    ("physics_off_topic",     "relevance"):        (0.01, "Completely off-topic — about cooking."),
    # is_voice=False

    ("math_markdown_response","hallucination"):    (0.88, "Formula is correct; notation is different."),
    ("math_markdown_response","correctness"):      (0.92, "Formula is correct."),
    ("math_markdown_response","relevance"):        (0.95, "Directly answers the question."),
    ("math_markdown_response","audio_clarity"):    (0.06, "LaTeX, markdown headers and bold make this unreadable by TTS."),
}


# ── Helpers ───────────────────────────────────────────────────────────────────

@dataclass
class CaseResult:
    case_id: str
    scores: dict = field(default_factory=dict)  # metric → EvalScore


def _make_mock_client(score: float, reasoning: str, score_key: str):
    payload = {score_key: score, "reasoning": reasoning, "tts_issues": []}
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": json.dumps(payload)}}]
    }
    mock_client = AsyncMock()
    mock_client.__aenter__.return_value = mock_client
    mock_client.__aexit__.return_value = None
    mock_client.post.return_value = mock_resp
    return mock_client


# ── Main suite test ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_full_eval_suite(override_settings):
    """
    Run all evaluation metrics over every case in EVAL_DATASET.
    Prints a full score table — no test fails on score values.
    """
    results: List[CaseResult] = []

    for case in EVAL_DATASET:
        cr = CaseResult(case_id=case["id"])

        # ── hallucination ──
        s, r = _MOCK_SCORES[(case["id"], "hallucination")]
        with patch("httpx.AsyncClient", return_value=_make_mock_client(s, r, "hallucination_score")):
            cr.scores["hallucination"] = await judge_hallucination(
                case["question"], case["context"], case["response"], api_key="test-key"
            )

        # ── correctness ──
        s, r = _MOCK_SCORES[(case["id"], "correctness")]
        with patch("httpx.AsyncClient", return_value=_make_mock_client(s, r, "correctness_score")):
            cr.scores["correctness"] = await judge_correctness(
                case["question"], case["reference"], case["response"], api_key="test-key"
            )

        # ── relevance ──
        s, r = _MOCK_SCORES[(case["id"], "relevance")]
        with patch("httpx.AsyncClient", return_value=_make_mock_client(s, r, "relevance_score")):
            cr.scores["relevance"] = await judge_relevance(
                case["question"], case["response"], api_key="test-key"
            )

        # ── socratic quality (socratic agent only) ──
        if case["agent"] == "socratic":
            s, r = _MOCK_SCORES[(case["id"], "socratic_quality")]
            with patch("httpx.AsyncClient", return_value=_make_mock_client(s, r, "socratic_quality_score")):
                cr.scores["socratic_quality"] = await judge_socratic_quality(
                    case["question"], case["response"], api_key="test-key"
                )

        # ── audio clarity (voice-enabled cases only) ──
        if case["is_voice"]:
            s, r = _MOCK_SCORES[(case["id"], "audio_clarity")]
            with patch("httpx.AsyncClient", return_value=_make_mock_client(s, r, "audio_clarity_score")):
                cr.scores["audio_clarity"] = await judge_audio_clarity(
                    case["response"], api_key="test-key"
                )

        results.append(cr)

    # ── Print score table ──────────────────────────────────────────────────────
    col_case   = 30
    col_metric = 20
    col_score  = 8

    header_line = (
        f"\n{'CASE ID':<{col_case}} "
        f"{'METRIC':<{col_metric}} "
        f"{'SCORE':>{col_score}}   REASONING"
    )
    separator = "-" * 100

    print("\n")
    print("=" * 100)
    print("  EDUMENTOR EVALUATION RESULTS")
    print("=" * 100)
    print(header_line)
    print(separator)

    prev_case = None
    for cr in results:
        if prev_case and prev_case != cr.case_id:
            print()
        prev_case = cr.case_id
        for metric, es in cr.scores.items():
            bar = _score_bar(es.score)
            print(
                f"{cr.case_id:<{col_case}} "
                f"{metric:<{col_metric}} "
                f"{es.score:>{col_score}.2f}  {bar}  {es.reasoning[:60]}"
            )

    print("=" * 100)

    # ── Aggregate summary ─────────────────────────────────────────────────────
    all_scores = [es for cr in results for es in cr.scores.values()]
    avg_by_metric: dict[str, list[float]] = {}
    for es in all_scores:
        avg_by_metric.setdefault(es.metric, []).append(es.score)

    print("\n  AVERAGE SCORES BY METRIC")
    print(separator)
    for metric, scores_list in sorted(avg_by_metric.items()):
        avg = sum(scores_list) / len(scores_list)
        bar = _score_bar(avg)
        print(f"  {metric:<{col_metric}} avg={avg:.2f}  {bar}  (n={len(scores_list)})")
    print("=" * 100 + "\n")

    # ── Only structural assertions — no score gates ────────────────────────────
    for es in all_scores:
        assert 0.0 <= es.score <= 1.0, f"Out-of-range score for {es.metric}: {es.score}"
        assert es.reasoning, f"Empty reasoning for {es.metric}"


def _score_bar(score: float, width: int = 10) -> str:
    """Visual bar: ████████░░ for a score of 0.8."""
    filled = round(score * width)
    return "█" * filled + "░" * (width - filled)


# ── Per-metric structural tests ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_all_metrics_return_evalscores(override_settings):
    """Every judge function must return an EvalScore with valid fields."""
    funcs = [
        (judge_hallucination,   {"question": "Q", "context": "C", "response": "R"}, "hallucination_score"),
        (judge_correctness,     {"question": "Q", "reference": "Ref", "response": "R"}, "correctness_score"),
        (judge_relevance,       {"question": "Q", "response": "R"}, "relevance_score"),
        (judge_socratic_quality,{"question": "Q", "response": "R"}, "socratic_quality_score"),
        (judge_audio_clarity,   {"response": "R"}, "audio_clarity_score"),
    ]
    for fn, kwargs, score_key in funcs:
        with patch("httpx.AsyncClient", return_value=_make_mock_client(0.75, "Test reasoning.", score_key)):
            result = await fn(**kwargs, api_key="test-key")

        assert isinstance(result, EvalScore),        f"{fn.__name__} did not return EvalScore"
        assert isinstance(result.score, float),      f"{fn.__name__} score is not float"
        assert 0.0 <= result.score <= 1.0,           f"{fn.__name__} score out of range"
        assert isinstance(result.reasoning, str),    f"{fn.__name__} reasoning is not str"
        assert isinstance(result.raw_response, str), f"{fn.__name__} raw_response is not str"
        assert result.metric != "",                  f"{fn.__name__} metric is empty"
