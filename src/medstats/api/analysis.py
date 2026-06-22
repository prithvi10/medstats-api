from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from uuid import UUID
from src.shared.db.session import get_db
from src.shared.security.auth import get_current_user
from src.medstats.db.models import (
    Submission, AnalysisJob, SubmissionStatus, JobStatus, UserRole
)

router = APIRouter()


@router.post("/submissions/{submission_id}/run")
async def trigger_analysis(
    submission_id: UUID,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Trigger the analysis pipeline for a submission."""
    result = await db.execute(select(Submission).where(Submission.id == submission_id))
    submission = result.scalar_one_or_none()
    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found")

    if current_user["role"] not in (UserRole.PROFESSOR.value, UserRole.ADMIN.value):
        if str(submission.student_id) != current_user["sub"]:
            raise HTTPException(status_code=403, detail="Not authorized")

    job = AnalysisJob(
        submission_id=submission_id,
        status=JobStatus.QUEUED,
        current_step="Queued for processing",
    )
    db.add(job)
    submission.status = SubmissionStatus.PARSING
    await db.commit()
    await db.refresh(job)

    # Trigger Celery task
    from src.medstats.services.worker import run_analysis_pipeline
    run_analysis_pipeline.delay(str(job.id))

    return {
        "job_id": str(job.id),
        "status": job.status.value,
        "message": "Analysis pipeline triggered",
    }


@router.get("/submissions/{submission_id}/job")
async def get_job_status(
    submission_id: UUID,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get the latest analysis job status for a submission."""
    result = await db.execute(
        select(AnalysisJob)
        .where(AnalysisJob.submission_id == submission_id)
        .order_by(AnalysisJob.created_at.desc())
    )
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="No analysis job found")

    return {
        "job_id": str(job.id),
        "status": job.status.value,
        "progress": job.progress,
        "current_step": job.current_step,
        "error_log": job.error_log,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
    }


@router.post("/submissions/{submission_id}/rerun")
async def rerun_analysis(
    submission_id: UUID,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Re-run analysis after corrections."""
    if current_user["role"] not in (UserRole.PROFESSOR.value, UserRole.ADMIN.value):
        raise HTTPException(status_code=403, detail="Only professors can trigger reruns")

    result = await db.execute(select(Submission).where(Submission.id == submission_id))
    submission = result.scalar_one_or_none()
    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found")

    job = AnalysisJob(
        submission_id=submission_id,
        status=JobStatus.QUEUED,
        current_step="Re-analysis queued",
    )
    db.add(job)
    submission.status = SubmissionStatus.ANALYZING
    await db.commit()
    await db.refresh(job)

    from src.medstats.services.worker import run_analysis_pipeline
    run_analysis_pipeline.delay(str(job.id))

    return {
        "job_id": str(job.id),
        "status": job.status.value,
        "message": "Re-analysis triggered",
    }
