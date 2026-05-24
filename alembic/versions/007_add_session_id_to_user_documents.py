"""Add session_id to user_documents to scope doc retrieval per session.

Without this column, has_user_docs and retrieve_user_docs had no session
boundary — a doc uploaded in session A would bleed into every future session
for the same user.

Revision ID: 007
Revises: 006
Create Date: 2026-05-22
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

revision = "007"
down_revision = "006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Nullable so existing rows (uploaded before this migration) are not broken.
    # Rows without a session_id will simply never be returned by the new
    # session-scoped has_user_docs query — correct behaviour, they're stale.
    op.add_column(
        "user_documents",
        sa.Column("session_id", UUID(as_uuid=True), nullable=True),
    )
    op.create_index(
        "ix_user_documents_session_id",
        "user_documents",
        ["session_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_user_documents_session_id", table_name="user_documents")
    op.drop_column("user_documents", "session_id")