import uuid
from datetime import datetime, timezone
from sqlalchemy import (
    Column, String, Text, DateTime, Enum, ForeignKey, Integer, JSON, Boolean
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from src.shared.db.session import Base
import enum


class UserRole(str, enum.Enum):
    STUDENT = "student"
    PROFESSOR = "professor"
    ADMIN = "admin"


class SubmissionStatus(str, enum.Enum):
    SUBMITTED = "submitted"
    PARSING = "parsing"
    ANALYZING = "analyzing"
    REPORT_READY = "report_ready"
    UNDER_REVIEW = "under_review"
    REVISION = "revision"
    APPROVED = "approved"
    EXPORTED = "exported"
    FAILED = "failed"


class JobStatus(str, enum.Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ReviewDecision(str, enum.Enum):
    APPROVED = "approved"
    NEEDS_REVISION = "needs_revision"
    REJECTED = "rejected"


class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    name = Column(String(255), nullable=False)
    role = Column(Enum(UserRole), nullable=False)
    institution = Column(String(255), nullable=True)
    department = Column(String(255), nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    submissions = relationship("Submission", back_populates="student", foreign_keys="Submission.student_id")
    reviews = relationship("ProfessorReview", back_populates="professor")


class ProfessorStudent(Base):
    __tablename__ = "professor_students"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    professor_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    student_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    academic_year = Column(String(20), nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    professor = relationship("User", foreign_keys=[professor_id])
    student = relationship("User", foreign_keys=[student_id])


class Submission(Base):
    __tablename__ = "submissions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    student_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    professor_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    title = Column(String(500), nullable=False)
    intent_description = Column(Text, nullable=True)
    status = Column(Enum(SubmissionStatus), default=SubmissionStatus.SUBMITTED)
    metadata_ = Column("metadata", JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    student = relationship("User", back_populates="submissions", foreign_keys=[student_id])
    professor = relationship("User", foreign_keys=[professor_id])
    files = relationship("SubmissionFile", back_populates="submission", cascade="all, delete-orphan")
    jobs = relationship("AnalysisJob", back_populates="submission", cascade="all, delete-orphan")
    reports = relationship("Report", back_populates="submission", cascade="all, delete-orphan")


class SubmissionFile(Base):
    __tablename__ = "submission_files"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    submission_id = Column(UUID(as_uuid=True), ForeignKey("submissions.id"), nullable=False)
    file_url = Column(String(1000), nullable=False)
    file_type = Column(String(50), nullable=False)  # data_excel, data_csv, intent_doc, supporting_doc
    original_name = Column(String(500), nullable=False)
    file_size = Column(Integer, nullable=True)
    parsed_metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    submission = relationship("Submission", back_populates="files")


class AnalysisJob(Base):
    __tablename__ = "analysis_jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    submission_id = Column(UUID(as_uuid=True), ForeignKey("submissions.id"), nullable=False)
    status = Column(Enum(JobStatus), default=JobStatus.QUEUED)
    agent_state = Column(JSON, nullable=True)
    test_results = Column(JSON, nullable=True)
    charts_metadata = Column(JSON, nullable=True)
    progress = Column(Integer, default=0)  # 0-100
    current_step = Column(String(255), nullable=True)
    error_log = Column(Text, nullable=True)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    submission = relationship("Submission", back_populates="jobs")
    report = relationship("Report", back_populates="analysis_job", uselist=False)


class Report(Base):
    __tablename__ = "reports"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    submission_id = Column(UUID(as_uuid=True), ForeignKey("submissions.id"), nullable=False)
    analysis_job_id = Column(UUID(as_uuid=True), ForeignKey("analysis_jobs.id"), nullable=True)
    version = Column(Integer, default=1)
    sections = Column(JSON, nullable=True)  # Ordered list of report sections
    charts = Column(JSON, nullable=True)  # Chart metadata + URLs
    summary = Column(Text, nullable=True)
    methodology_note = Column(Text, nullable=True)
    pdf_url = Column(String(1000), nullable=True)
    docx_url = Column(String(1000), nullable=True)
    generated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    submission = relationship("Submission", back_populates="reports")
    analysis_job = relationship("AnalysisJob", back_populates="report")
    reviews = relationship("ProfessorReview", back_populates="report", cascade="all, delete-orphan")


class ProfessorReview(Base):
    __tablename__ = "professor_reviews"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    report_id = Column(UUID(as_uuid=True), ForeignKey("reports.id"), nullable=False)
    professor_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    decision = Column(Enum(ReviewDecision), nullable=False)
    feedback = Column(Text, nullable=True)
    section_comments = Column(JSON, nullable=True)  # {section_index: comment}
    reviewed_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    report = relationship("Report", back_populates="reviews")
    professor = relationship("User", back_populates="reviews")
    corrections = relationship("AgentCorrection", back_populates="review", cascade="all, delete-orphan")


class AgentCorrection(Base):
    __tablename__ = "agent_corrections"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    submission_id = Column(UUID(as_uuid=True), ForeignKey("submissions.id"), nullable=False)
    review_id = Column(UUID(as_uuid=True), ForeignKey("professor_reviews.id"), nullable=False)
    correction_prompt = Column(Text, nullable=False)
    changes_made = Column(JSON, nullable=True)
    new_report_id = Column(UUID(as_uuid=True), ForeignKey("reports.id"), nullable=True)
    corrected_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    review = relationship("ProfessorReview", back_populates="corrections")
