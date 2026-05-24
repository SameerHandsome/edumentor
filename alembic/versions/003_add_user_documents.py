"""Add user_documents table for tracking per-user uploaded files.

Revision ID: 003
Revises: 002
Create Date: 2025-01-03
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, TIMESTAMP

revision = "003"
down_revision = "002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "user_documents",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "user_id",
            UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        # doc_id matches the Qdrant point group key — used for deletion
        sa.Column("doc_id", sa.String(64), nullable=False, unique=True),
        sa.Column("filename", sa.String(500), nullable=False),
        sa.Column("chunk_count", sa.Integer, server_default="0", nullable=False),
        sa.Column("created_at", TIMESTAMP(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    # Fast lookup by user — drives the "list my documents" query
    op.create_index("ix_user_documents_user_id", "user_documents", ["user_id"])


def downgrade() -> None:
    op.drop_index("ix_user_documents_user_id", table_name="user_documents")
    op.drop_table("user_documents")
