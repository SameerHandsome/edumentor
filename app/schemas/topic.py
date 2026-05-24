"""Topic schemas."""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel


class TopicResponse(BaseModel):
    id: UUID
    name: str
    slug: str
    description: str
    parent_id: UUID | None
    grade_level: int
    order_index: int

    model_config = {"from_attributes": True}


class TopicListResponse(BaseModel):
    topics: list[TopicResponse]
    total: int
