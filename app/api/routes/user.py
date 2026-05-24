"""User profile and mastery routes."""

from __future__ import annotations

from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.routes.deps import get_current_user_id
from app.core.database import get_db
from app.models.mastery import MasteryProfile
from app.models.topic import Topic
from app.models.user import User, UserPreference
from app.schemas.user import MasteryResponse, UserProfileResponse, UserProfileUpdate

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/user", tags=["user"])


@router.get("/profile", response_model=UserProfileResponse)
async def get_profile(
    user_id: str = Depends(get_current_user_id), db: AsyncSession = Depends(get_db)
):
    result = await db.execute(select(User).where(User.id == UUID(user_id)))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    pref_result = await db.execute(
        select(UserPreference).where(UserPreference.user_id == UUID(user_id))
    )
    pref = pref_result.scalar_one_or_none()
    return UserProfileResponse.model_validate(
        {**user.__dict__, "preferences": pref.__dict__ if pref else None}
    )


@router.patch("/profile", response_model=UserProfileResponse)
async def update_profile(
    body: UserProfileUpdate,
    user_id: str = Depends(get_current_user_id),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(User).where(User.id == UUID(user_id)))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if body.full_name:
        user.full_name = body.full_name
    pref_result = await db.execute(
        select(UserPreference).where(UserPreference.user_id == UUID(user_id))
    )
    pref = pref_result.scalar_one_or_none()
    if pref:
        if body.explanation_style:
            pref.explanation_style = body.explanation_style
        if body.session_goal is not None:
            pref.session_goal = body.session_goal
        if body.weak_topics is not None:
            pref.weak_topics = body.weak_topics
    await db.flush()
    return await get_profile(user_id=user_id, db=db)


@router.get("/mastery", response_model=list[MasteryResponse])
async def get_mastery(
    user_id: str = Depends(get_current_user_id), db: AsyncSession = Depends(get_db)
):
    result = await db.execute(
        select(MasteryProfile, Topic)
        .join(Topic, MasteryProfile.topic_id == Topic.id)
        .where(MasteryProfile.user_id == UUID(user_id))
    )
    rows = result.all()
    return [
        MasteryResponse(
            topic_id=mp.topic_id,
            topic_name=t.name,
            theta=mp.theta,
            attempts=mp.attempts,
            correct=mp.correct,
            accuracy=mp.correct / mp.attempts if mp.attempts else 0.0,
        )
        for mp, t in rows
    ]


@router.get("/mastery/{topic_id}", response_model=MasteryResponse)
async def get_topic_mastery(
    topic_id: UUID, user_id: str = Depends(get_current_user_id), db: AsyncSession = Depends(get_db)
):
    result = await db.execute(
        select(MasteryProfile, Topic)
        .join(Topic, MasteryProfile.topic_id == Topic.id)
        .where(MasteryProfile.user_id == UUID(user_id), MasteryProfile.topic_id == topic_id)
    )
    row = result.one_or_none()
    if not row:
        raise HTTPException(status_code=404, detail="Mastery profile not found")
    mp, t = row
    return MasteryResponse(
        topic_id=mp.topic_id,
        topic_name=t.name,
        theta=mp.theta,
        attempts=mp.attempts,
        correct=mp.correct,
        accuracy=mp.correct / mp.attempts if mp.attempts else 0.0,
    )
