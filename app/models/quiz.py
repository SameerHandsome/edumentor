"""Quiz question and attempt models."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class QuizQuestion(Base):
    __tablename__ = "quiz_questions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    topic_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("topics.id", ondelete="CASCADE"), nullable=False, index=True
    )
    question_text: Mapped[str] = mapped_column(Text, nullable=False)
    choices: Mapped[dict] = mapped_column(
        JSONB, nullable=False
    )  # {"A": "...", "B": "...", "C": "...", "D": "..."}
    correct_answer: Mapped[str] = mapped_column(String(1), nullable=False)  # A|B|C|D
    explanation: Mapped[str] = mapped_column(Text, default="")
    difficulty_b: Mapped[float] = mapped_column(Float, default=0.0)  # IRT b-parameter
    discrimination_a: Mapped[float] = mapped_column(Float, default=1.0)  # IRT a-parameter
    guessing_c: Mapped[float] = mapped_column(Float, default=0.25)  # IRT c-parameter (3PL)
    created_by: Mapped[str] = mapped_column(String(50), default="dspy")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )

    attempts: Mapped[list[QuizAttempt]] = relationship("QuizAttempt", back_populates="question")


class QuizAttempt(Base):
    __tablename__ = "quiz_attempts"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False
    )
    question_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("quiz_questions.id", ondelete="CASCADE"), nullable=False
    )
    selected_answer: Mapped[str] = mapped_column(String(1), nullable=False)
    is_correct: Mapped[bool] = mapped_column(Boolean, nullable=False)
    theta_before: Mapped[float] = mapped_column(Float, default=0.0)
    theta_after: Mapped[float] = mapped_column(Float, default=0.0)
    time_taken_seconds: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )

    question: Mapped[QuizQuestion] = relationship("QuizQuestion", back_populates="attempts")
