"""Initial EduMentor schema.

Revision ID: 001
Revises: —
Create Date: 2025-01-01
"""
from __future__ import annotations
import uuid
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB, TIMESTAMP

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table("users",
        sa.Column("id", UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column("email", sa.String(255), nullable=False, unique=True),
        sa.Column("hashed_password", sa.String(255), nullable=False),
        sa.Column("full_name", sa.String(255), nullable=False),
        sa.Column("is_active", sa.Boolean, default=True),
        sa.Column("is_verified", sa.Boolean, default=False),
        sa.Column("created_at", TIMESTAMP(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", TIMESTAMP(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_users_email", "users", ["email"])

    op.create_table("user_preferences",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("explanation_style", sa.String(50), default="step_by_step"),
        sa.Column("preferred_language", sa.String(10), default="en"),
        sa.Column("session_goal", sa.Text, default=""),
        sa.Column("weak_topics", JSONB, default=list),
        sa.Column("extra", JSONB, default=dict),
        sa.Column("updated_at",TIMESTAMP(timezone=True), server_default=sa.func.now()),
    )

    op.create_table("topics",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("slug", sa.String(255), nullable=False, unique=True),
        sa.Column("description", sa.Text,default=""),
        sa.Column("parent_id", UUID(as_uuid=True), sa.ForeignKey("topics.id", ondelete="SET NULL"), nullable=True),
        sa.Column("grade_level", sa.Integer, default=10),
        sa.Column("order_index", sa.Integer, default=0),
        sa.Column("metadata", JSONB, default=dict),
    )
    op.create_index("ix_topics_slug", "topics", ["slug"])

    op.create_table("mastery_profiles",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("topic_id", UUID(as_uuid=True), sa.ForeignKey("topics.id", ondelete="CASCADE"), nullable=False),
        sa.Column("theta", sa.Float, default=0.0),
        sa.Column("attempts", sa.Integer, default=0),
        sa.Column("correct", sa.Integer, default=0),
        sa.Column("updated_at", TIMESTAMP(timezone=True), server_default=sa.func.now()),
    )

    op.create_table("sessions",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("topic_id", UUID(as_uuid=True), sa.ForeignKey("topics.id", ondelete="SET NULL"), nullable=True),
        sa.Column("started_at", TIMESTAMP(timezone=True), server_default=sa.func.now()),
        sa.Column("ended_at", TIMESTAMP(timezone=True), nullable=True),
        sa.Column("is_active", sa.Boolean, default=True),
        sa.Column("agent_type", sa.String(50), default="socratic"),
        sa.Column("summary", sa.Text, default=""),
        sa.Column("metadata", JSONB, default=dict),
    )

    op.create_table("messages",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("session_id", UUID(as_uuid=True), sa.ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("role", sa.String(20), nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("agent_type", sa.String(50), default=""),
        sa.Column("langsmith_trace_id", sa.String(255), default=""),
        sa.Column("created_at", TIMESTAMP(timezone=True), server_default=sa.func.now()),
        sa.Column("metadata", JSONB, default=dict),
    )

    op.create_table("quiz_questions",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("topic_id", UUID(as_uuid=True), sa.ForeignKey("topics.id", ondelete="CASCADE"), nullable=False),
        sa.Column("question_text", sa.Text, nullable=False),
        sa.Column("choices", JSONB, nullable=False),
        sa.Column("correct_answer", sa.String(1), nullable=False),
        sa.Column("explanation", sa.Text, default=""),
        sa.Column("difficulty_b", sa.Float, default=0.0),
        sa.Column("discrimination_a", sa.Float, default=1.0),
        sa.Column("guessing_c", sa.Float, default=0.25),
        sa.Column("created_by", sa.String(50), default="dspy"),
        sa.Column("created_at", TIMESTAMP(timezone=True), server_default=sa.func.now()),
    )

    op.create_table("quiz_attempts",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("session_id", UUID(as_uuid=True), sa.ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("question_id", UUID(as_uuid=True), sa.ForeignKey("quiz_questions.id", ondelete="CASCADE"), nullable=False),
        sa.Column("selected_answer", sa.String(1), nullable=False),
        sa.Column("is_correct", sa.Boolean, nullable=False),
        sa.Column("theta_before", sa.Float, default=0.0),
        sa.Column("theta_after", sa.Float, default=0.0),
        sa.Column("time_taken_seconds", sa.Integer, default=0),
        sa.Column("created_at", TIMESTAMP(timezone=True), server_default=sa.func.now()),
    )

    op.create_table("feedback",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("session_id", UUID(as_uuid=True), sa.ForeignKey("sessions.id", ondelete="SET NULL"), nullable=True),
        sa.Column("message_id", UUID(as_uuid=True), sa.ForeignKey("messages.id", ondelete="SET NULL"), nullable=True),
        sa.Column("langsmith_trace_id", sa.String(255), default=""),
        sa.Column("rating", sa.Integer, nullable=False),
        sa.Column("comment", sa.Text, default=""),
        sa.Column("is_dpo_candidate", sa.Boolean, default=False),
        sa.Column("created_at",TIMESTAMP(timezone=True), server_default=sa.func.now()),
    )

    op.create_table("jobs",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", UUID(as_uuid=True), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("job_type", sa.String(50), nullable=False),
        sa.Column("celery_task_id", sa.String(255), default=""),
        sa.Column("status", sa.String(30), default="pending"),
        sa.Column("result", JSONB, default=dict),
        sa.Column("error", sa.Text, default=""),
        sa.Column("created_at", TIMESTAMP(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", TIMESTAMP(timezone=True), server_default=sa.func.now()),
    )


def downgrade() -> None:
    for table in ["jobs", "feedback", "quiz_attempts", "quiz_questions",
                  "messages", "sessions", "mastery_profiles", "topics",
                  "user_preferences", "users"]:
        op.drop_table(table)
