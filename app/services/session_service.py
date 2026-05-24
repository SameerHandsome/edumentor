"""Business logic for tutor sessions."""

from __future__ import annotations

from datetime import UTC
from uuid import UUID

import structlog
from sqlalchemy import desc, select
from sqlalchemy.exc import InterfaceError, OperationalError
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.mastery import MasteryProfile
from app.models.session import Message, Session
from app.models.user import UserPreference

logger = structlog.get_logger(__name__)


async def create_session(db: AsyncSession, user_id: UUID, topic_id: UUID | None = None) -> Session:
    session = Session(user_id=user_id, topic_id=topic_id)
    db.add(session)
    await db.flush()
    return session


async def end_session(db: AsyncSession, session_id: UUID) -> Session | None:
    from datetime import datetime

    result = await db.execute(select(Session).where(Session.id == session_id))
    session = result.scalar_one_or_none()
    if session:
        session.is_active = False
        session.ended_at = datetime.now(UTC)
        await db.flush()
    return session


async def save_message(
    db: AsyncSession,
    session_id: UUID,
    role: str,
    content: str,
    agent_type: str = "",
    trace_id: str = "",
) -> Message:
    """Save a chat message, retrying once if the DB connection was closed (Neon idle drop)."""
    msg = Message(
        session_id=session_id,
        role=role,
        content=content,
        agent_type=agent_type,
        langsmith_trace_id=trace_id,
    )
    db.add(msg)
    try:
        await db.flush()
    except (InterfaceError, OperationalError):
        # Neon closed the idle connection — rollback, expire identity map, retry
        await db.rollback()
        db.expire_all()
        db.add(msg)
        await db.flush()
    return msg


async def get_session_history_from_db(
    db: AsyncSession, session_id: UUID, limit: int = 5
) -> list[dict]:
    result = await db.execute(
        select(Message)
        .where(Message.session_id == session_id)
        .order_by(desc(Message.created_at))
        .limit(limit)
    )
    messages = result.scalars().all()
    return [{"role": m.role, "content": m.content} for m in reversed(messages)]


async def get_user_preferences(db: AsyncSession, user_id: UUID) -> dict:
    result = await db.execute(select(UserPreference).where(UserPreference.user_id == user_id))
    pref = result.scalar_one_or_none()
    if not pref:
        return {
            "explanation_style": "step_by_step",
            "weak_topics": [],
            "session_goal": "",
            "language": "en",
        }
    return {
        "explanation_style": pref.explanation_style,
        "weak_topics": pref.weak_topics or [],
        "session_goal": pref.session_goal or "",
        "language": pref.preferred_language,
    }


async def get_user_mastery(db: AsyncSession, user_id: UUID, topic_id: UUID | None = None) -> dict:
    query = select(MasteryProfile).where(MasteryProfile.user_id == user_id)
    if topic_id:
        query = query.where(MasteryProfile.topic_id == topic_id)
    result = await db.execute(query)
    profiles = result.scalars().all()
    if not profiles:
        return {"theta": 0.0, "level": "intermediate"}
    avg_theta = sum(p.theta for p in profiles) / len(profiles)
    from app.services.irt import theta_to_level

    return {"theta": avg_theta, "level": theta_to_level(avg_theta)}
