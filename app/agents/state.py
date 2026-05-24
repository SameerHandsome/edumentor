"""LangGraph shared state schema."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class EduMentorState(BaseModel):
    """Shared state across all agent nodes in the LangGraph graph."""

    # Session context
    session_id: str = ""
    user_id: str = ""
    topic_id: str = ""  # UUID string — used for DB FK references
    topic_name: str = ""  # Human-readable label — used for memory filtering
    has_user_docs: bool = False  # True when user has uploaded docs in Qdrant

    # Student profile (from PostgreSQL)
    theta: float = 0.0
    student_level: str = "intermediate"
    explanation_style: str = "step_by_step"
    weak_topics: list[str] = Field(default_factory=list)
    session_goal: str = ""
    language: str = "en"

    # Current turn
    user_query: str = ""
    intent: str = ""  # socratic|quiz|explain|feedback|meta
    agent_response: str = ""
    agent_type: str = ""  # which agent produced the response
    langsmith_trace_id: str = ""

    # RAG context (populated by retrieval nodes)
    rag_chunks: list[str] = Field(default_factory=list)
    user_doc_chunks: list[str] = Field(default_factory=list)  # from user's own uploaded docs
    session_summary: str = ""

    # History (last 5, from Redis)
    history: list[dict[str, str]] = Field(default_factory=list)

    # Quiz state
    current_question_id: str = ""
    quiz_result: dict[str, Any] = Field(default_factory=dict)

    # Job tracking
    job_id: str = ""

    # Error propagation
    error: str = ""

    class Config:
        arbitrary_types_allowed = True
