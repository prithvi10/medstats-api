"""Initial schema: all base tables (users, submissions, jobs, reports, reviews).

Revision ID: 000_initial_schema
Revises:
Create Date: 2026-07-15
"""
from alembic import op
import sqlalchemy as sa

revision = "000_initial_schema"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    dialect = bind.dialect.name

    # Create enum types for PostgreSQL first
    if dialect == "postgresql":
        op.execute(
            """
            DO $$
            BEGIN
                IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'userrole') THEN
                    CREATE TYPE userrole AS ENUM ('student', 'professor', 'admin');
                END IF;
                IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'submissionstatus') THEN
                    CREATE TYPE submissionstatus AS ENUM (
                        'submitted', 'parsing', 'analyzing', 'report_ready',
                        'ai_reviewed', 'under_review', 'review_in_progress',
                        'review_completed', 'revision', 'approved', 'exported', 'failed'
                    );
                END IF;
                IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'jobstatus') THEN
                    CREATE TYPE jobstatus AS ENUM ('queued', 'running', 'completed', 'failed', 'cancelled');
                END IF;
                IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'reviewdecision') THEN
                    CREATE TYPE reviewdecision AS ENUM ('approve', 'revise', 'reject');
                END IF;
            END
            $$;
            """
        )

    op.create_table(
        "users",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("email", sa.String(255), nullable=False, unique=True, index=True),
        sa.Column("password_hash", sa.String(255), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("role", sa.Enum("student", "professor", "admin", name="userrole"), nullable=False),
        sa.Column("institution", sa.String(255), nullable=True),
        sa.Column("department", sa.String(255), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=True, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )

    op.create_table(
        "professor_students",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("professor_id", sa.String(36), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("student_id", sa.String(36), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("academic_year", sa.String(20), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
    )

    op.create_table(
        "submissions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("student_id", sa.String(36), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("professor_id", sa.String(36), sa.ForeignKey("users.id"), nullable=True),
        sa.Column("title", sa.String(500), nullable=False),
        sa.Column("intent_description", sa.Text(), nullable=True),
        sa.Column(
            "status",
            sa.Enum(
                "submitted", "parsing", "analyzing", "report_ready",
                "ai_reviewed", "under_review", "review_in_progress",
                "review_completed", "revision", "approved", "exported", "failed",
                name="submissionstatus",
            ),
            nullable=True,
        ),
        sa.Column("metadata", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )

    op.create_table(
        "submission_files",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("submission_id", sa.String(36), sa.ForeignKey("submissions.id"), nullable=False),
        sa.Column("file_url", sa.String(1000), nullable=False),
        sa.Column("file_type", sa.String(50), nullable=False),
        sa.Column("original_name", sa.String(500), nullable=False),
        sa.Column("file_size", sa.Integer(), nullable=True),
        sa.Column("parsed_metadata", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
    )

    op.create_table(
        "analysis_jobs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("submission_id", sa.String(36), sa.ForeignKey("submissions.id"), nullable=False),
        sa.Column(
            "status",
            sa.Enum("queued", "running", "completed", "failed", "cancelled", name="jobstatus"),
            nullable=True,
        ),
        sa.Column("agent_state", sa.JSON(), nullable=True),
        sa.Column("test_results", sa.JSON(), nullable=True),
        sa.Column("charts_metadata", sa.JSON(), nullable=True),
        sa.Column("progress", sa.Integer(), nullable=True),
        sa.Column("current_step", sa.String(255), nullable=True),
        sa.Column("error_log", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
    )

    op.create_table(
        "reports",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("submission_id", sa.String(36), sa.ForeignKey("submissions.id"), nullable=False),
        sa.Column("analysis_job_id", sa.String(36), sa.ForeignKey("analysis_jobs.id"), nullable=True),
        sa.Column("version", sa.Integer(), nullable=True),
        sa.Column("sections", sa.JSON(), nullable=True),
        sa.Column("charts", sa.JSON(), nullable=True),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("methodology_note", sa.Text(), nullable=True),
        sa.Column("pdf_url", sa.String(1000), nullable=True),
        sa.Column("docx_url", sa.String(1000), nullable=True),
        sa.Column("generated_at", sa.DateTime(timezone=True), nullable=True),
    )

    op.create_table(
        "professor_reviews",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("report_id", sa.String(36), sa.ForeignKey("reports.id"), nullable=False),
        sa.Column("professor_id", sa.String(36), sa.ForeignKey("users.id"), nullable=False),
        sa.Column(
            "decision",
            sa.Enum("approve", "revise", "reject", name="reviewdecision"),
            nullable=False,
        ),
        sa.Column("feedback", sa.Text(), nullable=True),
        sa.Column("section_comments", sa.JSON(), nullable=True),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
    )

    op.create_table(
        "agent_corrections",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("submission_id", sa.String(36), sa.ForeignKey("submissions.id"), nullable=False),
        sa.Column("review_id", sa.String(36), sa.ForeignKey("professor_reviews.id"), nullable=False),
        sa.Column("correction_prompt", sa.Text(), nullable=False),
        sa.Column("changes_made", sa.JSON(), nullable=True),
        sa.Column("new_report_id", sa.String(36), sa.ForeignKey("reports.id"), nullable=True),
        sa.Column("corrected_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("agent_corrections")
    op.drop_table("professor_reviews")
    op.drop_table("reports")
    op.drop_table("analysis_jobs")
    op.drop_table("submission_files")
    op.drop_table("submissions")
    op.drop_table("professor_students")
    op.drop_table("users")
