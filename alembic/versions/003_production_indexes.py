"""Add production performance indexes.

Revision ID: 003_production_indexes
Revises: 002_graphrag
Create Date: 2026-07-14
"""
from alembic import op

revision = "003_production_indexes"
down_revision = "002_graphrag"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    dialect = bind.dialect.name

    op.create_index("ix_submissions_status", "submissions", ["status"], unique=False)
    op.create_index("ix_submissions_student_id", "submissions", ["student_id"], unique=False)
    op.create_index("ix_jobs_submission_id", "analysis_jobs", ["submission_id"], unique=False)
    op.create_index("ix_jobs_status", "analysis_jobs", ["status"], unique=False)
    op.create_index("ix_notifs_user_read", "notifications", ["user_id", "is_read"], unique=False)

    if dialect == "postgresql":
        op.execute(
            """
            CREATE INDEX IF NOT EXISTS ix_lit_embedding_ivfflat
            ON literature_index USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100)
            """
        )


def downgrade() -> None:
    bind = op.get_bind()
    dialect = bind.dialect.name

    if dialect == "postgresql":
        op.execute("DROP INDEX IF EXISTS ix_lit_embedding_ivfflat")

    op.drop_index("ix_notifs_user_read", table_name="notifications")
    op.drop_index("ix_jobs_status", table_name="analysis_jobs")
    op.drop_index("ix_jobs_submission_id", table_name="analysis_jobs")
    op.drop_index("ix_submissions_student_id", table_name="submissions")
    op.drop_index("ix_submissions_status", table_name="submissions")
