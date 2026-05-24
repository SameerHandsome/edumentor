"""Curriculum topic routes."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.routes.deps import get_current_user_id
from app.core.database import get_db
from app.core.multi_layer_cache import ml_get_topics, ml_set_topics
from app.models.topic import Topic
from app.schemas.topic import TopicListResponse, TopicResponse

router = APIRouter(prefix="/curriculum", tags=["curriculum"])


@router.get("/topics", response_model=TopicListResponse)
async def list_topics(
    grade_level: int | None = Query(None),
    db: AsyncSession = Depends(get_db),
    _: str = Depends(get_current_user_id),
):
    # Only cache the unfiltered full topic list
    if grade_level is None:
        cached = await ml_get_topics()
        if cached:
            return cached

    query = select(Topic).where(Topic.parent_id is None).order_by(Topic.order_index)
    if grade_level:
        query = query.where(Topic.grade_level == grade_level)
    result = await db.execute(query)
    topics = result.scalars().all()
    response = TopicListResponse(
        topics=[TopicResponse.model_validate(t) for t in topics], total=len(topics)
    )

    if grade_level is None:
        await ml_set_topics(response.model_dump())

    return response


@router.get("/topics/{topic_id}", response_model=TopicResponse)
async def get_topic(
    topic_id: UUID, db: AsyncSession = Depends(get_db), _: str = Depends(get_current_user_id)
):
    result = await db.execute(select(Topic).where(Topic.id == topic_id))
    topic = result.scalar_one_or_none()
    if not topic:
        raise HTTPException(status_code=404, detail="Topic not found")
    return TopicResponse.model_validate(topic)


@router.get("/topics/{topic_id}/subtopics", response_model=TopicListResponse)
async def get_subtopics(
    topic_id: UUID, db: AsyncSession = Depends(get_db), _: str = Depends(get_current_user_id)
):
    result = await db.execute(
        select(Topic).where(Topic.parent_id == topic_id).order_by(Topic.order_index)
    )
    subtopics = result.scalars().all()
    return TopicListResponse(
        topics=[TopicResponse.model_validate(t) for t in subtopics], total=len(subtopics)
    )
