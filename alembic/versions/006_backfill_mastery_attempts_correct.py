"""Backfill NULL attempts and correct in mastery_profiles and enforce NOT NULL.

Revision ID: 006
Revises: 005
Create Date: 2026-05-16
"""
from alembic import op
import sqlalchemy as sa

revision = "006"
down_revision = "005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Backfill any existing NULLs left by rows inserted before server defaults existed
    op.execute("UPDATE mastery_profiles SET attempts = 0 WHERE attempts IS NULL")
    op.execute("UPDATE mastery_profiles SET correct  = 0 WHERE correct  IS NULL")

    # Add server defaults so future direct-SQL inserts also get 0, not NULL
    op.alter_column(
        "mastery_profiles",
        "attempts",
        existing_type=sa.Integer(),
        nullable=False,
        server_default="0",
    )
    op.alter_column(
        "mastery_profiles",
        "correct",
        existing_type=sa.Integer(),
        nullable=False,
        server_default="0",
    )


def downgrade() -> None:
    op.alter_column(
        "mastery_profiles",
        "attempts",
        existing_type=sa.Integer(),
        nullable=True,
        server_default=None,
    )
    op.alter_column(
        "mastery_profiles",
        "correct",
        existing_type=sa.Integer(),
        nullable=True,
        server_default=None,
    )