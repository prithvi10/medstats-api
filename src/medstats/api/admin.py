from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.medstats.db.models import (
    AgentCorrection,
    AnalysisJob,
    JobStatus,
    Submission,
    SubmissionStatus,
    User,
    UserRole,
)
from src.medstats.db.rag_models import KnowledgeNode, LiteratureIndex, RAGDocument
from src.shared.config.settings import get_settings
from src.shared.db.session import get_db
from src.shared.security.auth import get_current_user, hash_password

router = APIRouter()
settings = get_settings()


def _require_admin(current_user: dict):
    if current_user.get("role") != UserRole.ADMIN.value:
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user


class UserCreateRequest(BaseModel):
    email: str
    password: str = Field(min_length=6)
    name: str
    role: UserRole = UserRole.STUDENT
    institution: Optional[str] = None
    department: Optional[str] = None


class UserUpdateRequest(BaseModel):
    is_active: Optional[bool] = None
    role: Optional[UserRole] = None
    name: Optional[str] = None
    institution: Optional[str] = None
    department: Optional[str] = None


class SubmissionStatusUpdate(BaseModel):
    status: SubmissionStatus


def _job_to_dict(job: AnalysisJob, submission: Optional[Submission] = None, student: Optional[User] = None) -> dict:
    return {
        "id": str(job.id),
        "submission_id": str(job.submission_id),
        "submission_title": submission.title if submission else None,
        "student_name": student.name if student else None,
        "student_email": student.email if student else None,
        "status": job.status.value,
        "progress": job.progress,
        "current_step": job.current_step,
        "error_log": job.error_log,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
        "created_at": job.created_at.isoformat() if job.created_at else None,
    }


@router.get("/dashboard")
async def admin_dashboard(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    _require_admin(current_user)

    total_users = await db.execute(select(func.count(User.id)))
    total_submissions = await db.execute(select(func.count(Submission.id)))
    total_jobs = await db.execute(select(func.count(AnalysisJob.id)))

    students = await db.execute(select(func.count(User.id)).where(User.role == UserRole.STUDENT))
    professors = await db.execute(select(func.count(User.id)).where(User.role == UserRole.PROFESSOR))
    running_jobs = await db.execute(
        select(func.count(AnalysisJob.id)).where(AnalysisJob.status == JobStatus.RUNNING)
    )
    failed_jobs = await db.execute(
        select(func.count(AnalysisJob.id)).where(AnalysisJob.status == JobStatus.FAILED)
    )

    return {
        "total_users": total_users.scalar(),
        "total_submissions": total_submissions.scalar(),
        "total_jobs": total_jobs.scalar(),
        "students_count": students.scalar(),
        "professors_count": professors.scalar(),
        "jobs_running": running_jobs.scalar(),
        "jobs_failed": failed_jobs.scalar(),
    }


@router.get("/users")
async def list_users(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    _require_admin(current_user)
    result = await db.execute(select(User).order_by(User.created_at.desc()))
    users = result.scalars().all()
    return {
        "users": [
            {
                "id": str(u.id),
                "email": u.email,
                "name": u.name,
                "role": u.role.value,
                "institution": u.institution,
                "department": u.department,
                "is_active": u.is_active,
                "created_at": u.created_at.isoformat(),
            }
            for u in users
        ]
    }


@router.get("/users/{user_id}")
async def get_user(
    user_id: UUID,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    _require_admin(current_user)
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    sub_count = await db.execute(
        select(func.count(Submission.id)).where(Submission.student_id == user.id)
    )
    return {
        "id": str(user.id),
        "email": user.email,
        "name": user.name,
        "role": user.role.value,
        "institution": user.institution,
        "department": user.department,
        "is_active": user.is_active,
        "created_at": user.created_at.isoformat(),
        "submission_count": sub_count.scalar(),
    }


@router.post("/users")
async def create_user(
    body: UserCreateRequest,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    _require_admin(current_user)
    existing = await db.execute(select(User).where(User.email == body.email))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Email already registered")

    user = User(
        email=body.email,
        password_hash=hash_password(body.password),
        name=body.name,
        role=body.role,
        institution=body.institution,
        department=body.department,
        is_active=True,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return {"id": str(user.id), "email": user.email, "role": user.role.value}


@router.patch("/users/{user_id}")
async def update_user(
    user_id: UUID,
    body: UserUpdateRequest,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    _require_admin(current_user)
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if body.is_active is not None:
        user.is_active = body.is_active
    if body.role is not None:
        user.role = body.role
    if body.name is not None:
        user.name = body.name
    if body.institution is not None:
        user.institution = body.institution
    if body.department is not None:
        user.department = body.department

    await db.commit()
    return {"id": str(user.id), "is_active": user.is_active, "role": user.role.value}


@router.get("/jobs")
async def list_jobs(
    status: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    _require_admin(current_user)

    query = (
        select(AnalysisJob, Submission, User)
        .join(Submission, Submission.id == AnalysisJob.submission_id)
        .join(User, User.id == Submission.student_id)
        .order_by(AnalysisJob.created_at.desc())
    )
    if status:
        try:
            job_status = JobStatus(status)
            query = query.where(AnalysisJob.status == job_status)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid status: {status}")

    count_q = select(func.count()).select_from(query.subquery())
    total = (await db.execute(count_q)).scalar() or 0

    offset = (page - 1) * per_page
    result = await db.execute(query.offset(offset).limit(per_page))
    rows = result.all()

    return {
        "jobs": [_job_to_dict(job, sub, student) for job, sub, student in rows],
        "total": total,
        "page": page,
        "per_page": per_page,
    }


@router.get("/jobs/{job_id}")
async def get_job(
    job_id: UUID,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    _require_admin(current_user)
    result = await db.execute(
        select(AnalysisJob, Submission, User)
        .join(Submission, Submission.id == AnalysisJob.submission_id)
        .join(User, User.id == Submission.student_id)
        .where(AnalysisJob.id == job_id)
    )
    row = result.first()
    if not row:
        raise HTTPException(status_code=404, detail="Job not found")

    job, sub, student = row
    corrections = await db.execute(
        select(AgentCorrection).where(AgentCorrection.submission_id == job.submission_id)
        .order_by(AgentCorrection.corrected_at.desc())
    )

    data = _job_to_dict(job, sub, student)
    data["agent_state"] = job.agent_state
    data["test_results"] = job.test_results
    data["charts_metadata"] = job.charts_metadata
    data["corrections"] = [
        {
            "id": str(c.id),
            "review_id": str(c.review_id),
            "changes_made": c.changes_made,
            "corrected_at": c.corrected_at.isoformat() if c.corrected_at else None,
        }
        for c in corrections.scalars().all()
    ]
    return data


@router.post("/jobs/{job_id}/retry")
async def retry_job(
    job_id: UUID,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    _require_admin(current_user)
    result = await db.execute(select(AnalysisJob).where(AnalysisJob.id == job_id))
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status not in (JobStatus.FAILED, JobStatus.CANCELLED):
        raise HTTPException(status_code=400, detail="Only failed or cancelled jobs can be retried")

    job.status = JobStatus.QUEUED
    job.progress = 0
    job.current_step = "Queued for processing"
    job.error_log = None
    job.started_at = None
    job.completed_at = None
    await db.commit()

    if settings.ENV == "production":
        from src.medstats.services.worker import run_analysis_pipeline
        run_analysis_pipeline.delay(str(job.id))
    else:
        from src.medstats.api.analysis import _run_pipeline_inline
        background_tasks.add_task(_run_pipeline_inline, str(job.id))

    return {"message": "Job requeued", "job_id": str(job.id)}


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(
    job_id: UUID,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    _require_admin(current_user)
    result = await db.execute(select(AnalysisJob).where(AnalysisJob.id == job_id))
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != JobStatus.RUNNING:
        raise HTTPException(status_code=400, detail="Only running jobs can be cancelled")

    job.status = JobStatus.FAILED
    job.error_log = "Cancelled by admin"
    job.current_step = "Cancelled"
    job.completed_at = datetime.now(timezone.utc)
    await db.commit()
    return {"message": "Job cancelled", "job_id": str(job.id)}


@router.get("/submissions")
async def list_submissions(
    status: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    _require_admin(current_user)

    query = (
        select(Submission, User)
        .join(User, User.id == Submission.student_id)
        .order_by(Submission.updated_at.desc())
    )
    if status:
        try:
            sub_status = SubmissionStatus(status)
            query = query.where(Submission.status == sub_status)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid status: {status}")

    if search:
        like = f"%{search}%"
        query = query.where(or_(Submission.title.ilike(like), User.email.ilike(like), User.name.ilike(like)))

    count_q = select(func.count()).select_from(query.subquery())
    total = (await db.execute(count_q)).scalar() or 0
    offset = (page - 1) * per_page
    result = await db.execute(query.offset(offset).limit(per_page))
    rows = result.all()

    return {
        "submissions": [
            {
                "id": str(sub.id),
                "title": sub.title,
                "status": sub.status.value,
                "student_name": user.name,
                "student_email": user.email,
                "created_at": sub.created_at.isoformat(),
                "updated_at": sub.updated_at.isoformat(),
            }
            for sub, user in rows
        ],
        "total": total,
        "page": page,
        "per_page": per_page,
    }


@router.patch("/submissions/{submission_id}/status")
async def update_submission_status(
    submission_id: UUID,
    body: SubmissionStatusUpdate,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    _require_admin(current_user)
    result = await db.execute(select(Submission).where(Submission.id == submission_id))
    submission = result.scalar_one_or_none()
    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found")

    submission.status = body.status
    await db.commit()
    return {"id": str(submission.id), "status": submission.status.value}


async def _redis_ping() -> bool:
    try:
        import redis
        r = redis.from_url(settings.REDIS_URL, socket_connect_timeout=2)
        return r.ping()
    except Exception:
        return False


async def _queue_depth() -> int:
    try:
        import redis
        r = redis.from_url(settings.REDIS_URL, socket_connect_timeout=2)
        return int(r.llen("celery") or 0)
    except Exception:
        return -1


@router.get("/system/health")
async def system_health(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    _require_admin(current_user)

    db_ok = True
    try:
        await db.execute(select(func.count(User.id)))
    except Exception:
        db_ok = False

    redis_ok = await _redis_ping()
    queue_depth = await _queue_depth() if redis_ok else -1

    users = (await db.execute(select(func.count(User.id)))).scalar()
    submissions = (await db.execute(select(func.count(Submission.id)))).scalar()
    running = (await db.execute(
        select(func.count(AnalysisJob.id)).where(AnalysisJob.status == JobStatus.RUNNING)
    )).scalar()
    failed = (await db.execute(
        select(func.count(AnalysisJob.id)).where(AnalysisJob.status == JobStatus.FAILED)
    )).scalar()

    return {
        "db_ok": db_ok,
        "redis_ok": redis_ok,
        "queue_depth": queue_depth,
        "counts": {
            "users": users,
            "submissions": submissions,
            "jobs_running": running,
            "jobs_failed": failed,
        },
    }


@router.get("/rag/stats")
async def rag_stats(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    _require_admin(current_user)

    doc_count = (await db.execute(select(func.count(RAGDocument.id)))).scalar() or 0
    lit_count = (await db.execute(select(func.count(LiteratureIndex.id)))).scalar() or 0
    node_count = (await db.execute(select(func.count(KnowledgeNode.id)))).scalar() or 0

    with_embedding = (await db.execute(
        select(func.count(RAGDocument.id)).where(RAGDocument.embedding.isnot(None))
    )).scalar() or 0
    coverage = round((with_embedding / doc_count * 100) if doc_count else 0, 1)

    return {
        "doc_count": doc_count,
        "literature_count": lit_count,
        "knowledge_nodes": node_count,
        "embedding_coverage_pct": coverage,
    }


async def _run_pubmed_ingest():
    from src.shared.db.session import async_session_factory
    from src.medstats.rag.corpus.pubmed import PubMedCorpusIndexer

    async with async_session_factory() as session:
        indexer = PubMedCorpusIndexer(session)
        total = await indexer.index_all_disciplines(max_per_discipline=100)
        await session.commit()
        return total


def _run_pubmed_ingest_sync():
    asyncio.run(_run_pubmed_ingest())


@router.post("/rag/ingest")
async def trigger_rag_ingest(
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
):
    _require_admin(current_user)
    background_tasks.add_task(_run_pubmed_ingest_sync)
    return {"message": "PubMed ingest started in background"}
