"""Add notifications table and new submission statuses.

Revision ID: 001_notifications
Revises:
Create Date: 2026-07-12
"""
from alembic import op
import sqlalchemy as sa

revision = "001_notifications"
down_revision = "000_initial_schema"
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
                    "submitted", "parsing", "analyzing", "report_ready",
                    "ai_reviewed", "under_review", "review_in_progress",
                    "review_completed", "revision", "approved", "exported", "failed",
                    name="submissionstatus",
                ),
                type_=sa.Enum(
                    "submitted", "parsing", "analyzing", "report_ready",
                    "ai_reviewed", "under_review", "review_in_progress",
                    "review_completed", "revision", "approved", "exported", "failed",
                    name="submissionstatus",
                ),
            )
    else:
        # Add any enum values not present in older installs (000 already creates them on fresh DBs)
        op.execute(
            """
            DO $$
            BEGIN
                IF NOT EXISTS (SELECT 1 FROM pg_enum WHERE enumlabel = 'ai_reviewed'
                               AND enumtypid = 'submissionstatus'::regtype) THEN
                    ALTER TYPE submissionstatus ADD VALUE 'ai_reviewed';
                END IF;
                IF NOT EXISTS (SELECT 1 FROM pg_enum WHERE enumlabel = 'review_in_progress'
                               AND enumtypid = 'submissionstatus'::regtype) THEN
                    ALTER TYPE submissionstatus ADD VALUE 'review_in_progress';
                END IF;
                IF NOT EXISTS (SELECT 1 FROM pg_enum WHERE enumlabel = 'review_completed'
                               AND enumtypid = 'submissionstatus'::regtype) THEN
                    ALTER TYPE submissionstatus ADD VALUE 'review_completed';
                END IF;
            END
            $$;
            """
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
