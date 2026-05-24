"""Backfill NULL theta in mastery_profiles to 0.0 and enforce NOT NULL.

Revision ID: 005
Revises: 004
Create Date: 2026-05-16
"""
from alembic import op
import sqlalchemy as sa

revision = "005"
down_revision = "004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1. Backfill any existing NULLs
    op.execute("UPDATE mastery_profiles SET theta = 0.0 WHERE theta IS NULL")
    # 2. Set server default so future inserts without a value get 0.0
    op.alter_column(
        "mastery_profiles",
        "theta",
        existing_type=sa.Float(),
        nullable=False,
        server_default="0.0",
    )


def downgrade() -> None:
    op.alter_column(
        "mastery_profiles",
        "theta",
        existing_type=sa.Float(),
        nullable=True,
        server_default=None,
    )