"""Quiz schemas."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class QuizRequestSchema(BaseModel):
    session_id: UUID
    topic_id: UUID
    num_questions: int = 5


class QuizSubmitSchema(BaseModel):
    session_id: UUID
    question_id: UUID
    selected_answer: str
    time_taken_seconds: int = 0


class QuizQuestionOut(BaseModel):
    id: UUID
    question_text: str
    choices: dict[str, str]
    difficulty_b: float


class QuizResultResponse(BaseModel):
    question_id: UUID
    is_correct: bool
    correct_answer: str
    explanation: str
    theta_before: float
    theta_after: float


class QuizHistoryItem(BaseModel):
    id: UUID
    session_id: UUID
    is_correct: bool
    created_at: datetime

    model_config = {"from_attributes": True}
