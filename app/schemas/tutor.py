"""Tutor session schemas."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class StartSessionRequest(BaseModel):
    topic_id: UUID | None = None
    session_goal: str | None = None


class StartSessionResponse(BaseModel):
    session_id: UUID
    message: str = "Session started"


class TextRequest(BaseModel):
    session_id: UUID
    message: str = ""


class VoiceResponse(BaseModel):
    job_id: UUID
    status: str = "processing"


class TextResponse(BaseModel):
    session_id: UUID
    reply: str
    agent_type: str | None = None
    trace_id: str | None = None
    message_id: UUID | None = None  # DB id of the saved assistant Message row
    quiz_redirect: bool = False
    topic_id: str | None = None
    topic_name: str | None = None  # extracted from message when topic_id is unavailable


class SessionResponse(BaseModel):
    id: UUID
    topic_id: UUID | None
    started_at: datetime
    ended_at: datetime | None
    is_active: bool
    agent_type: str
    display_name: str | None = None  # stored in metadata_["display_name"]

    model_config = {"from_attributes": True}

    @classmethod
    def from_session(cls, s) -> SessionResponse:
        meta = s.metadata_ or {}
        return cls(
            id=s.id,
            topic_id=s.topic_id,
            started_at=s.started_at,
            ended_at=s.ended_at,
            is_active=s.is_active,
            agent_type=s.agent_type,
            display_name=meta.get("display_name"),
        )


class MessageResponse(BaseModel):
    id: UUID
    role: str
    content: str
    agent_type: str
    created_at: datetime

    model_config = {"from_attributes": True}


class FeedbackRequest(BaseModel):
    session_id: UUID
    message_id: UUID | None = None
    langsmith_trace_id: str = ""
    rating: int
    comment: str = ""


class JobStatusResponse(BaseModel):
    job_id: UUID
    status: str
    result: dict | None = None
    error: str = ""
