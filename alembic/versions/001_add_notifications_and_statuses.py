"""Add notifications table and new submission statuses.

Revision ID: 001_notifications
Revises:
Create Date: 2026-07-12
"""
from alembic import op
import sqlalchemy as sa

revision = "001_notifications"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    dialect = bind.dialect.name

    if dialect == "sqlite":
        with op.batch_alter_table("submissions") as batch_op:
            batch_op.alter_column(
                "status",
                existing_type=sa.Enum(
                    "submitted",
                    "parsing",
                    "analyzing",
                    "report_ready",
                    "AI_REVIEWED",
                    "under_review",
                    "REVIEW_IN_PROGRESS",
                    "REVIEW_COMPLETED",
                    "revision",
                    "approved",
                    "exported",
                    "failed",
                    name="submissionstatus",
                ),
                type_=sa.Enum(
                    "submitted",
                    "parsing",
                    "analyzing",
                    "report_ready",
                    "AI_REVIEWED",
                    "under_review",
                    "REVIEW_IN_PROGRESS",
                    "REVIEW_COMPLETED",
                    "revision",
                    "approved",
                    "exported",
                    "failed",
                    name="submissionstatus",
                ),
            )
    else:
        op.execute(
            "ALTER TYPE submissionstatus ADD VALUE IF NOT EXISTS 'AI_REVIEWED'"
        )
        op.execute(
            "ALTER TYPE submissionstatus ADD VALUE IF NOT EXISTS 'REVIEW_IN_PROGRESS'"
        )
        op.execute(
            "ALTER TYPE submissionstatus ADD VALUE IF NOT EXISTS 'REVIEW_COMPLETED'"
        )

    op.create_table(
        "notifications",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("user_id", sa.String(length=36), nullable=False),
        sa.Column("type", sa.String(length=50), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("message", sa.Text(), nullable=True),
        sa.Column("submission_id", sa.String(length=36), nullable=True),
        sa.Column("is_read", sa.Boolean(), nullable=True, server_default=sa.false()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["submission_id"], ["submissions.id"]),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_notifications_user_id", "notifications", ["user_id"])


def downgrade() -> None:
    op.drop_index("ix_notifications_user_id", table_name="notifications")
    op.drop_table("notifications")
