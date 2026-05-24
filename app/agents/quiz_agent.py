"""
Quiz Generator Agent — DSPy ChainOfThought with IRT difficulty targeting.
Generates MCQ matched to |b - theta| < 0.5 difficulty window.

LM backend: Groq (meta-llama/llama-4-scout-17b-16e-instruct)
  — fast (2-5 s), reliable structured output, no asyncpg timeout risk.

DSPy usage:
  - Offline: run `python -m scripts.optimize_quiz_dspy` once to compile and save
             the optimized module to app/agents/dspy_optimized_quiz.json
  - Runtime: loads the saved module automatically — zero extra cost.
  - Fallback: if no saved module found, runs unoptimized ChainOfThought.

Generation priority:
  1. DSPy ChainOfThought via Groq (optimized if dspy_optimized_quiz.json exists)
  2. Direct Groq API call (httpx, no DSPy/LiteLLM) — true independent fallback

Note: theta and b_target are passed as strings to DSPy (DSPy 2.5 requires
all InputField values to be str). Conversion happens in _generate_via_dspy.
"""

from __future__ import annotations

import asyncio
import json
import os
import re

import httpx
import structlog
from langsmith import traceable

from app.agents.state import EduMentorState
from app.core.config import settings

logger = structlog.get_logger(__name__)

OPTIMIZED_MODULE_PATH = os.path.join(os.path.dirname(__file__), "dspy_optimized_quiz.json")

# ── Groq endpoint constants ───────────────────────────────────────────────────
_GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
# Strip the "groq/" prefix that DSPy uses — raw httpx needs just the model name
_GROQ_MODEL_NAME = settings.QUIZ_LM_MODEL.removeprefix("groq/")

try:
    import dspy

    _DSPY_AVAILABLE = True
except ImportError:
    _DSPY_AVAILABLE = False
    logger.warning("dspy_not_installed", fallback="groq_direct")


# ── DSPy Signature ────────────────────────────────────────────────────────────


class QuizSignature(dspy.Signature if _DSPY_AVAILABLE else object):  # type: ignore[misc]
    """Generate a multiple-choice quiz question at a specific IRT difficulty level."""

    # All fields must be str — DSPy 2.5 requires string inputs
    topic: str = dspy.InputField(desc="The topic for the question") if _DSPY_AVAILABLE else None
    theta: str = (
        dspy.InputField(desc="Student ability as IRT theta float, e.g. '0.50'")
        if _DSPY_AVAILABLE
        else None
    )
    b_target: str = (
        dspy.InputField(desc="Target difficulty b-parameter float, e.g. '0.70'")
        if _DSPY_AVAILABLE
        else None
    )
    question: str = (
        dspy.OutputField(desc="Question text only — no JSON, no markdown")
        if _DSPY_AVAILABLE
        else None
    )
    choice_a: str = dspy.OutputField(desc="Choice A text only") if _DSPY_AVAILABLE else None
    choice_b: str = dspy.OutputField(desc="Choice B text only") if _DSPY_AVAILABLE else None
    choice_c: str = dspy.OutputField(desc="Choice C text only") if _DSPY_AVAILABLE else None
    choice_d: str = dspy.OutputField(desc="Choice D text only") if _DSPY_AVAILABLE else None
    correct: str = (
        dspy.OutputField(desc="Correct answer letter: A, B, C, or D") if _DSPY_AVAILABLE else None
    )
    explanation: str = (
        dspy.OutputField(desc="Brief explanation of the correct answer")
        if _DSPY_AVAILABLE
        else None
    )


# ── DSPy generator (lazy singleton) ──────────────────────────────────────────


def _get_dspy_generator():
    """
    Lazy-initialize DSPy with Groq as the LM backend.
    Loads optimized module from disk if available.
    Falls back to unoptimized ChainOfThought if no saved module found.
    """
    if not _DSPY_AVAILABLE:
        return None

    if not settings.GROQ_API_KEY:
        logger.error("groq_api_key_missing", hint="Set GROQ_API_KEY in .env")
        return None

    lm = dspy.LM(
        model=settings.QUIZ_LM_MODEL,  # "groq/meta-llama/llama-4-scout-17b-16e-instruct"
        api_key=settings.GROQ_API_KEY,
        max_tokens=settings.DSPY_MAX_TOKENS,
        temperature=0.6,
    )
    dspy.configure(lm=lm)

    module = dspy.ChainOfThought(QuizSignature)

    if os.path.exists(OPTIMIZED_MODULE_PATH):
        try:
            module.load(OPTIMIZED_MODULE_PATH)
            logger.info("dspy_optimized_module_loaded", path=OPTIMIZED_MODULE_PATH)
        except Exception as exc:
            logger.warning("dspy_load_failed", error=str(exc), fallback="unoptimized_cot")
    else:
        logger.warning(
            "dspy_no_optimized_module",
            path=OPTIMIZED_MODULE_PATH,
            hint="Run: python -m scripts.optimize_quiz_dspy",
        )

    return module


_dspy_generator = None


# ── Output validation ─────────────────────────────────────────────────────────


def _is_valid_question(data: dict) -> bool:
    """
    Reject garbage before it reaches the DB.
    Checks:
      - question is a non-empty plain string (not raw JSON / markdown fences)
      - exactly 4 choices present
      - correct_answer is one of A/B/C/D
    """
    question = data.get("question", "").strip()
    if not question:
        return False
    if question.startswith(("{", "`", "[")):
        return False
    choices = data.get("choices", {})
    if len(choices) != 4:
        return False
    if data.get("correct_answer", "") not in ("A", "B", "C", "D"):
        return False
    return True


# ── Tier 1: DSPy + Groq ───────────────────────────────────────────────────────


async def _generate_via_dspy(topic: str, theta: float, b_target: float) -> dict:
    """Generate a question using DSPy ChainOfThought with Groq as the LM."""
    global _dspy_generator
    if _dspy_generator is None:
        _dspy_generator = _get_dspy_generator()
    if _dspy_generator is None:
        return {}

    try:
        theta_str = f"{theta:.2f}"
        b_target_str = f"{b_target:.2f}"
        loop = asyncio.get_event_loop()

        # Attempt 1
        try:
            result = await loop.run_in_executor(
                None,
                lambda: _dspy_generator(topic=topic, theta=theta_str, b_target=b_target_str),
            )
        except Exception as exc_1:
            logger.warning("dspy_attempt_1_failed", error=str(exc_1), hint="retrying once")
            # Attempt 2 — retry once; Groq is stateless so a second call is cheap
            result = await loop.run_in_executor(
                None,
                lambda: _dspy_generator(topic=topic, theta=theta_str, b_target=b_target_str),
            )

        data = {
            "question": result.question,
            "choices": {
                "A": result.choice_a,
                "B": result.choice_b,
                "C": result.choice_c,
                "D": result.choice_d,
            },
            "correct_answer": result.correct.strip().upper()[:1],
            "explanation": result.explanation,
            "difficulty_b": b_target,
        }

        if not _is_valid_question(data):
            logger.warning("dspy_invalid_output", topic=topic, data=str(data)[:200])
            return {}

        return data

    except Exception as exc:
        logger.error("dspy_quiz_failed", error=str(exc))
        return {}


# ── Tier 2: Direct Groq API (httpx, no DSPy/LiteLLM) ────────────────────────


async def _generate_via_groq_direct(
    topic: str,
    theta: float,
    b_target: float,
    exclude_questions: list[str] | None = None,
) -> dict:
    """
    True independent fallback — bypasses DSPy and LiteLLM entirely.
    Calls Groq's OpenAI-compatible endpoint directly via httpx.
    Returns a validated question dict or {} on failure.

    exclude_questions: list of already-generated question texts the LLM must
    avoid, so repeated calls on the same topic produce distinct questions.
    """
    if not settings.GROQ_API_KEY:
        logger.error("groq_api_key_missing_fallback")
        return {}

    system_prompt = (
        "You are a quiz question generator. "
        "Respond ONLY with a JSON object — no markdown fences, no preamble. "
        "Required keys: question (string), choices (object with keys A/B/C/D), "
        "correct_answer (one of: A, B, C, D), explanation (string)."
    )

    exclude_block = ""
    if exclude_questions:
        lines = "\n".join(f"- {q}" for q in exclude_questions[:20])
        exclude_block = f"\n\nDo NOT generate any of these questions (already used):\n{lines}"

    user_prompt = (
        f"Generate ONE multiple-choice question about: {topic}.\n"
        f"IRT difficulty b={b_target:.2f}, student ability theta={theta:.2f}.\n"
        f"Match difficulty to the student's level."
        f"{exclude_block}"
    )

    payload = {
        "model": _GROQ_MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.6,
        "max_tokens": settings.DSPY_MAX_TOKENS,
    }

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                _GROQ_API_URL,
                headers={
                    "Authorization": f"Bearer {settings.GROQ_API_KEY}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            response.raise_for_status()

        raw = response.json()["choices"][0]["message"]["content"].strip()

        # Strip markdown fences if model adds them despite instructions
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        data = json.loads(raw)

        # Normalize: model might return choices as a list instead of a dict
        choices = data.get("choices", {})
        if isinstance(choices, list) and len(choices) == 4:
            choices = {"A": choices[0], "B": choices[1], "C": choices[2], "D": choices[3]}
            data["choices"] = choices

        data["difficulty_b"] = b_target
        data["correct_answer"] = str(data.get("correct_answer", "A")).strip().upper()[:1]

        if not _is_valid_question(data):
            logger.warning("groq_direct_invalid_output", topic=topic, raw=raw[:200])
            return {}

        return data

    except (httpx.HTTPError, json.JSONDecodeError, KeyError) as exc:
        logger.error("groq_direct_failed", error=str(exc))
        return {}


# ── quiz_agent entry point ────────────────────────────────────────────────────


@traceable(name="quiz_generator", project_name="edumentor")
async def quiz_agent(state: EduMentorState) -> EduMentorState:
    """
    Generate an IRT-difficulty-matched MCQ for the student.
    Target: |b - theta| < settings.IRT_DIFFICULTY_WINDOW

    Generation priority:
      1. DSPy ChainOfThought via Groq  — primary
      2. Direct Groq httpx call        — independent fallback (no DSPy/LiteLLM)
    """
    b_target = state.theta + 0.2
    # topic_id holds a UUID string when called from generate_quiz — passing a UUID
    # to the LLM produces off-topic questions.  topic_name (set by the route) is
    # the human-readable label; fall back to topic_id only if truly absent.
    topic_name = getattr(state, "topic_name", None) or state.topic_id or "general knowledge"

    # ── Tier 1: DSPy + Groq ───────────────────────────────────────────────────
    question_data = await _generate_via_dspy(topic_name, state.theta, b_target)

    if question_data:
        logger.info("quiz_dspy_success", topic=topic_name, b_target=b_target, theta=state.theta)
        return state.model_copy(
            update={
                "agent_response": json.dumps(question_data),
                "agent_type": "quiz",
                "quiz_result": question_data,
            }
        )

    # ── Tier 2: Direct Groq API call ──────────────────────────────────────────
    logger.warning("quiz_dspy_failed_or_empty", topic=topic_name, fallback="groq_direct")
    question_data = await _generate_via_groq_direct(topic_name, state.theta, b_target)

    if question_data:
        logger.info(
            "quiz_groq_direct_success", topic=topic_name, b_target=b_target, theta=state.theta
        )
        return state.model_copy(
            update={
                "agent_response": json.dumps(question_data),
                "agent_type": "quiz",
                "quiz_result": question_data,
            }
        )

    # Both tiers failed — return empty so the route falls through to DB backfill
    logger.error("quiz_all_tiers_failed", topic=topic_name, theta=state.theta)
    return state.model_copy(
        update={
            "agent_response": "",
            "agent_type": "quiz",
            "quiz_result": {},
        }
    )
