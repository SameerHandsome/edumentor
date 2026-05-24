"""Add GitHub OAuth fields to users table.

Revision ID: 002
Revises: 001
Create Date: 2025-01-02
"""
from alembic import op
import sqlalchemy as sa

revision = "002"
down_revision = "001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # hashed_password becomes nullable — OAuth users have no password
    op.alter_column("users", "hashed_password", nullable=True)

    # GitHub OAuth fields
    op.add_column("users", sa.Column("github_id", sa.String(100), nullable=True))
    op.add_column("users", sa.Column("oauth_provider", sa.String(30), nullable=True))
    op.add_column("users", sa.Column("avatar_url", sa.String(500), server_default="", nullable=False))

    # Unique index on github_id for fast lookup
    op.create_index("ix_users_github_id", "users", ["github_id"], unique=True)


def downgrade() -> None:
    op.drop_index("ix_users_github_id", table_name="users")
    op.drop_column("users", "avatar_url")
    op.drop_column("users", "oauth_provider")
    op.drop_column("users", "github_id")
    op.alter_column("users", "hashed_password", nullable=False)
