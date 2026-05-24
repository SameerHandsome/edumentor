"""
IRT (Item Response Theory) — 3PL model.

3PL probability: P(θ) = c + (1-c) / (1 + exp(-a*(θ-b)))
  θ = student ability (theta)
  a = discrimination parameter
  b = difficulty parameter
  c = guessing parameter

Theta update: EAP (Expected A Posteriori) approximation via one-step Newton update.
"""

from __future__ import annotations

import math

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.mastery import MasteryProfile
from app.models.quiz import QuizAttempt, QuizQuestion

logger = structlog.get_logger(__name__)


def p3pl(theta: float, a: float, b: float, c: float) -> float:
    """3PL probability of correct response."""
    return c + (1.0 - c) / (1.0 + math.exp(-a * (theta - b)))


def update_theta(
    theta: float,
    is_correct: bool,
    a: float = 1.0,
    b: float = 0.0,
    c: float = 0.25,
    learning_rate: float = 0.3,
) -> float:
    """
    Newton-Raphson single step for theta update.
    Clamps result to [-4, 4] to prevent runaway estimates.
    """
    p = p3pl(theta, a, b, c)
    response = 1.0 if is_correct else 0.0
    # First derivative of log-likelihood
    numerator = a * (1.0 - c) * math.exp(-a * (theta - b))
    denominator = (1.0 + math.exp(-a * (theta - b))) ** 2
    dp_dtheta = numerator / (denominator + 1e-9)
    gradient = dp_dtheta * (response - p) / (p * (1.0 - p) + 1e-9)
    new_theta = theta + learning_rate * gradient
    return max(-4.0, min(4.0, new_theta))


def theta_to_level(theta: float) -> str:
    """Convert IRT theta to human-readable level."""
    if theta < -1.0:
        return "beginner"
    elif theta < 1.0:
        return "intermediate"
    else:
        return "advanced"


async def score_attempt(
    db: AsyncSession,
    *,
    user_id: str,
    question_id: str,
    session_id: str,
    selected_answer: str,
    time_taken_seconds: int,
) -> dict:
    """
    Score a quiz attempt, update theta in mastery_profiles, persist attempt.
    Returns result dict with is_correct, theta_before, theta_after, explanation.
    """
    import uuid
    from uuid import UUID

    # Load question
    q_result = await db.execute(select(QuizQuestion).where(QuizQuestion.id == UUID(question_id)))
    question: QuizQuestion | None = q_result.scalar_one_or_none()
    if not question:
        raise ValueError(f"Question {question_id} not found")

    is_correct = selected_answer.upper() == question.correct_answer.upper()

    # Load or create mastery profile
    mp_result = await db.execute(
        select(MasteryProfile).where(
            MasteryProfile.user_id == UUID(user_id),
            MasteryProfile.topic_id == question.topic_id,
        )
    )
    mastery: MasteryProfile | None = mp_result.scalar_one_or_none()
    if not mastery:
        mastery = MasteryProfile(user_id=UUID(user_id), topic_id=question.topic_id)
        db.add(mastery)

    theta_before: float = mastery.theta or 0.0
    theta_after = update_theta(
        theta_before,
        is_correct,
        a=question.discrimination_a,
        b=question.difficulty_b,
        c=question.guessing_c,
    )

    mastery.theta = theta_after
    mastery.attempts = (mastery.attempts or 0) + 1
    if is_correct:
        mastery.correct = (mastery.correct or 0) + 1

    # Persist attempt
    attempt = QuizAttempt(
        id=uuid.uuid4(),
        user_id=UUID(user_id),
        session_id=UUID(session_id),
        question_id=UUID(question_id),
        selected_answer=selected_answer.upper(),
        is_correct=is_correct,
        theta_before=theta_before,
        theta_after=theta_after,
        time_taken_seconds=time_taken_seconds,
    )
    db.add(attempt)
    await db.flush()

    logger.info(
        "attempt_scored",
        user_id=user_id,
        is_correct=is_correct,
        theta_before=round(theta_before, 3),
        theta_after=round(theta_after, 3),
    )

    return {
        "is_correct": is_correct,
        "correct_answer": question.correct_answer,
        "explanation": question.explanation,
        "theta_before": theta_before,
        "theta_after": theta_after,
        "new_level": theta_to_level(theta_after),
    }
