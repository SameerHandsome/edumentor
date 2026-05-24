"""
Metadata filter construction for Qdrant pre-filtering.

SECURITY CONTRACT
─────────────────
• curriculum_docs  — no user_id filter; all users can access shared knowledge.
  Callers may narrow by topic_id, difficulty, grade_level, etc.
• user_memory      — user_id filter is MANDATORY and always injected by
  build_user_memory_filter().  Callers CANNOT omit it.

Pre-filtering runs BEFORE vector similarity search, so only matching
payload documents are even considered for ANN — this is both a performance
win and the isolation boundary.
"""

from __future__ import annotations

from typing import Any

from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    Range,
)

# ── curriculum_docs filters ──────────────────────────────────────────────────


def build_curriculum_filter(
    *,
    topic_id: str | None = None,
    subtopic_ids: list[str] | None = None,
    difficulty: str | None = None,
    difficulties: list[str] | None = None,
    language: str | None = None,
    doc_type: str | None = None,
    doc_types: list[str] | None = None,
    grade_level_min: int | None = None,
    grade_level_max: int | None = None,
    doc_ids: list[str] | None = None,
    source: str | None = None,
) -> Filter | None:
    """
    Build a Qdrant Filter for curriculum_docs.
    All parameters are optional — returns None if no conditions apply
    (Qdrant will then search the entire collection).

    Parameters
    ----------
    topic_id        : exact match on topic_id field
    subtopic_ids    : any-of match for multiple subtopics
    difficulty      : exact match — "beginner"|"intermediate"|"advanced"
    difficulties    : any-of match for multiple difficulty levels
    language        : ISO 639-1 code, e.g. "en"
    doc_type        : exact match — "textbook"|"lecture"|"worksheet"|"reference"
    doc_types       : any-of match for multiple doc types
    grade_level_min : lower bound (inclusive) on grade_level integer field
    grade_level_max : upper bound (inclusive) on grade_level integer field
    doc_ids         : restrict to specific document IDs (for targeted retrieval)
    source          : exact match on source field
    """
    conditions: list[FieldCondition] = []

    if topic_id:
        conditions.append(FieldCondition(key="topic_id", match=MatchValue(value=topic_id)))

    if subtopic_ids:
        conditions.append(FieldCondition(key="subtopic_id", match=MatchAny(any=subtopic_ids)))

    if difficulty and not difficulties:
        conditions.append(FieldCondition(key="difficulty", match=MatchValue(value=difficulty)))
    elif difficulties:
        conditions.append(FieldCondition(key="difficulty", match=MatchAny(any=difficulties)))

    if language:
        conditions.append(FieldCondition(key="language", match=MatchValue(value=language)))

    if doc_type and not doc_types:
        conditions.append(FieldCondition(key="doc_type", match=MatchValue(value=doc_type)))
    elif doc_types:
        conditions.append(FieldCondition(key="doc_type", match=MatchAny(any=doc_types)))

    if grade_level_min is not None or grade_level_max is not None:
        range_kwargs: dict[str, Any] = {}
        if grade_level_min is not None:
            range_kwargs["gte"] = grade_level_min
        if grade_level_max is not None:
            range_kwargs["lte"] = grade_level_max
        conditions.append(FieldCondition(key="grade_level", range=Range(**range_kwargs)))

    if doc_ids:
        conditions.append(FieldCondition(key="doc_id", match=MatchAny(any=doc_ids)))

    if source:
        conditions.append(FieldCondition(key="source", match=MatchValue(value=source)))

    if not conditions:
        return None

    return Filter(must=conditions)


# ── user_memory filters ───────────────────────────────────────────────────────


def build_user_memory_filter(
    user_id: str,
    *,
    memory_type: str | None = None,
    memory_types: list[str] | None = None,
    topic: str | None = None,
    session_id: str | None = None,
) -> Filter:
    """
    Build a Qdrant Filter for user_memory.

    user_id is REQUIRED and always injected as the first must-condition —
    this guarantees one user can NEVER retrieve another user's memory,
    regardless of what other filters are passed.

    Parameters
    ----------
    user_id      : (REQUIRED) scopes the entire query to one user
    memory_type  : "session_summary" | "preference" | "weak_area"
    memory_types : any-of match for multiple types
    topic        : filter by topic string
    session_id   : restrict to a specific session's memories
    """
    # user_id is the immutable isolation boundary — always first
    conditions: list[FieldCondition] = [
        FieldCondition(key="user_id", match=MatchValue(value=user_id))
    ]

    if memory_type and not memory_types:
        conditions.append(FieldCondition(key="type", match=MatchValue(value=memory_type)))
    elif memory_types:
        conditions.append(FieldCondition(key="type", match=MatchAny(any=memory_types)))

    if topic:
        conditions.append(FieldCondition(key="topic", match=MatchValue(value=topic)))

    if session_id:
        conditions.append(FieldCondition(key="session_id", match=MatchValue(value=session_id)))

    return Filter(must=conditions)


# ── user_docs filters ─────────────────────────────────────────────────────────


def build_user_docs_filter(
    user_id: str,
    *,
    doc_id: str | None = None,
    filename: str | None = None,
    session_id: str | None = None,
) -> Filter:
    """
    Build a Qdrant Filter for user_docs.

    user_id is REQUIRED — guarantees one user can never retrieve
    another user's uploaded documents.

    session_id should ALWAYS be passed so docs from a previous session
    are never injected into the current one.  Omitting it causes cross-
    session bleed where a doc uploaded 10 sessions ago appears as context.
    """
    conditions: list[FieldCondition] = [
        FieldCondition(key="user_id", match=MatchValue(value=user_id))
    ]

    if session_id:
        conditions.append(FieldCondition(key="session_id", match=MatchValue(value=session_id)))

    if doc_id:
        conditions.append(FieldCondition(key="doc_id", match=MatchValue(value=doc_id)))

    if filename:
        conditions.append(FieldCondition(key="filename", match=MatchValue(value=filename)))

    return Filter(must=conditions)
