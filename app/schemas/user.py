"""User profile schemas."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, EmailStr


class UserPreferenceSchema(BaseModel):
    explanation_style: str = "step_by_step"
    preferred_language: str = "en"
    session_goal: str = ""
    weak_topics: list[str] = []


class UserProfileResponse(BaseModel):
    id: UUID
    email: EmailStr
    full_name: str
    avatar_url: str = ""
    is_active: bool
    oauth_provider: str | None = None  # "github" or None (password auth)
    created_at: datetime
    preferences: UserPreferenceSchema | None = None

    model_config = {"from_attributes": True}


class UserProfileUpdate(BaseModel):
    full_name: str | None = None
    explanation_style: str | None = None
    session_goal: str | None = None
    weak_topics: list[str] | None = None


class MasteryResponse(BaseModel):
    topic_id: UUID
    topic_name: str
    theta: float
    attempts: int
    correct: int
    accuracy: float

    model_config = {"from_attributes": True}
