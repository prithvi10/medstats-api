"""
Analysis API routes.
Supports both Celery (production) and inline (local dev) pipeline execution.
"""
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from uuid import UUID
from pydantic import BaseModel
from typing import Optional
from src.shared.db.session import get_db
from src.shared.security.auth import get_current_user
from src.shared.config.settings import get_settings
from src.medstats.db.models import (
    Submission, AnalysisJob, SubmissionStatus, JobStatus, UserRole
)

router = APIRouter()
settings = get_settings()


class CorrectionRequest(BaseModel):
    feedback: str
    feedback_type: str = "general_feedback"
    target_variables: list = []


class ReviewRequest(BaseModel):
    """Request to run AI review on a completed analysis."""
    review_type: str = "full"  # full, mechanical_only, methodology_only


@router.post("/submissions/{submission_id}/run")
async def trigger_analysis(
    submission_id: UUID,
    background_tasks: BackgroundTasks,
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

    # Use Celery in production, BackgroundTasks locally
    if settings.ENV == "production":
        from src.medstats.services.worker import run_analysis_pipeline
        run_analysis_pipeline.delay(str(job.id))
    else:
        background_tasks.add_task(_run_pipeline_inline, str(job.id))

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
    background_tasks: BackgroundTasks,
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

    if settings.ENV == "production":
        from src.medstats.services.worker import run_analysis_pipeline
        run_analysis_pipeline.delay(str(job.id))
    else:
        background_tasks.add_task(_run_pipeline_inline, str(job.id))

    return {
        "job_id": str(job.id),
        "status": job.status.value,
        "message": "Re-analysis triggered",
    }


@router.post("/submissions/{submission_id}/correct")
async def apply_correction(
    submission_id: UUID,
    request: CorrectionRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Apply professor correction feedback to a report."""
    if current_user["role"] not in (UserRole.PROFESSOR.value, UserRole.ADMIN.value):
        raise HTTPException(status_code=403, detail="Only professors can submit corrections")

    result = await db.execute(select(Submission).where(Submission.id == submission_id))
    submission = result.scalar_one_or_none()
    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found")

    job = AnalysisJob(
        submission_id=submission_id,
        status=JobStatus.QUEUED,
        current_step="Correction in progress",
    )
    db.add(job)
    submission.status = SubmissionStatus.ANALYZING
    await db.commit()
    await db.refresh(job)

    if settings.ENV == "production":
        from src.medstats.services.worker import run_correction_pipeline
        run_correction_pipeline.delay(str(job.id), request.dict())
    else:
        background_tasks.add_task(_run_correction_inline, str(job.id), request.dict())

    return {
        "job_id": str(job.id),
        "status": "processing",
        "message": "Correction feedback submitted",
    }


@router.post("/submissions/{submission_id}/review")
async def run_ai_review(
    submission_id: UUID,
    request: ReviewRequest = ReviewRequest(),
    background_tasks: BackgroundTasks = None,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Run the AI Professor Review Agent on a completed analysis.

    This provides an automated quality check that can:
    - Replace manual professor review (AI-only mode)
    - Pre-screen before professor review (hybrid mode, reduces review time ~80%)
    """
    result = await db.execute(select(Submission).where(Submission.id == submission_id))
    submission = result.scalar_one_or_none()
    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found")

    # Get the latest completed job
    job_result = await db.execute(
        select(AnalysisJob)
        .where(AnalysisJob.submission_id == submission_id)
        .where(AnalysisJob.status == JobStatus.COMPLETED)
        .order_by(AnalysisJob.completed_at.desc())
    )
    completed_job = job_result.scalar_one_or_none()
    if not completed_job or not completed_job.result_data:
        raise HTTPException(status_code=404, detail="No completed analysis found to review")

    # Create a review job
    review_job = AnalysisJob(
        submission_id=submission_id,
        status=JobStatus.QUEUED,
        current_step="AI review queued",
    )
    db.add(review_job)
    await db.commit()
    await db.refresh(review_job)

    if settings.ENV == "production":
        from src.medstats.services.worker import run_review_pipeline
        run_review_pipeline.delay(str(review_job.id), str(completed_job.id), request.review_type)
    else:
        background_tasks.add_task(
            _run_review_inline, str(review_job.id), str(completed_job.id), request.review_type
        )

    return {
        "job_id": str(review_job.id),
        "status": "processing",
        "message": "AI Professor Review initiated",
    }


@router.get("/submissions/{submission_id}/review")
async def get_review_results(
    submission_id: UUID,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get the latest AI review results for a submission."""
    job_result = await db.execute(
        select(AnalysisJob)
        .where(AnalysisJob.submission_id == submission_id)
        .where(AnalysisJob.status == JobStatus.COMPLETED)
        .order_by(AnalysisJob.completed_at.desc())
    )
    job = job_result.scalar_one_or_none()
    if not job or not job.result_data:
        raise HTTPException(status_code=404, detail="No review results found")

    import json
    result_data = json.loads(job.result_data) if isinstance(job.result_data, str) else job.result_data

    review_summary = result_data.get("review_summary")
    if not review_summary:
        raise HTTPException(status_code=404, detail="No AI review in latest completed job")

    return {
        "submission_id": str(submission_id),
        "review_summary": review_summary,
        "review_verdict": result_data.get("review_verdict", "pending"),
    }


@router.get("/submissions/{submission_id}/export/{format}")
async def export_report(
    submission_id: UUID,
    format: str,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Export the generated report as DOCX or PDF."""
    if format not in ("docx", "pdf"):
        raise HTTPException(status_code=400, detail="Format must be 'docx' or 'pdf'")

    result = await db.execute(select(Submission).where(Submission.id == submission_id))
    submission = result.scalar_one_or_none()
    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found")

    # Get the latest completed job for this submission
    job_result = await db.execute(
        select(AnalysisJob)
        .where(AnalysisJob.submission_id == submission_id)
        .where(AnalysisJob.status == JobStatus.COMPLETED)
        .order_by(AnalysisJob.completed_at.desc())
    )
    job = job_result.scalar_one_or_none()
    if not job or not job.result_data:
        raise HTTPException(status_code=404, detail="No completed analysis found")

    import json
    report_data = json.loads(job.result_data) if isinstance(job.result_data, str) else job.result_data

    from src.medstats.services.export_service import export_report_to_docx, export_report_to_pdf

    export_fn = export_report_to_docx if format == "docx" else export_report_to_pdf
    filepath = export_fn(
        report_sections=report_data.get("report_sections", []),
        charts=report_data.get("charts", []),
        title=report_data.get("title", submission.title or "Report"),
        methodology_note=report_data.get("methodology_note", ""),
        summary=report_data.get("summary", ""),
        submission_id=str(submission_id),
    )

    media_type = (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        if format == "docx" else "application/pdf"
    )
    return FileResponse(filepath, media_type=media_type, filename=os.path.basename(filepath))


async def _run_pipeline_inline(job_id: str):
    """Run the analysis pipeline inline (for local dev without Celery)."""
    from src.shared.db.session import async_session_factory
    from src.medstats.agents.document_parser import DocumentParserAgent
    from src.medstats.agents.test_selector import TestSelectorAgent
    from src.medstats.agents.report_generator import ReportGeneratorAgent
    from src.agentic_core.agents.base_agent import MissionState
    from src.agentic_core.agents.orchestrator import MissionControl
    from datetime import datetime
    import json

    async with async_session_factory() as db:
        result = await db.execute(select(AnalysisJob).where(AnalysisJob.id == job_id))
        job = result.scalar_one_or_none()
        if not job:
            return

        job.status = JobStatus.RUNNING
        job.started_at = datetime.utcnow()
        await db.commit()

        try:
            # Build state from submission files
            sub_result = await db.execute(select(Submission).where(Submission.id == job.submission_id))
            submission = sub_result.scalar_one_or_none()

            state = MissionState(
                submission_id=str(job.submission_id),
                job_id=job_id,
            )
            state.agent_outputs["title"] = submission.title or ""
            state.agent_outputs["intent"] = submission.intent or ""
            state.agent_outputs["files"] = []  # TODO: load from SubmissionFile records

            agents = [
                DocumentParserAgent(services=None),
                TestSelectorAgent(services=None),
                ReportGeneratorAgent(services=None),
            ]
            orchestrator = MissionControl(agents=agents)

            async for updated_state in orchestrator.execute(state):
                job.progress = updated_state.progress
                job.current_step = f"Agent {updated_state.current_agent_index + 1}/3"
                await db.commit()

            # Run AI Review as post-pipeline step
            from src.medstats.agents.review_agent import ReviewAgent
            if updated_state.status == "COMPLETED":
                reviewer = ReviewAgent(services=None)
                updated_state = await reviewer.run(updated_state)

            if updated_state.status == "COMPLETED":
                job.status = JobStatus.COMPLETED
                job.progress = 100
                job.result_data = json.dumps({
                    "report_sections": updated_state.agent_outputs.get("report_sections", []),
                    "charts": updated_state.agent_outputs.get("charts", []),
                    "test_results": updated_state.agent_outputs.get("test_results", []),
                    "title": updated_state.agent_outputs.get("title", ""),
                    "methodology_note": updated_state.agent_outputs.get("methodology_note", ""),
                    "summary": updated_state.agent_outputs.get("summary", ""),
                    "review_summary": updated_state.agent_outputs.get("review_summary"),
                    "review_verdict": updated_state.agent_outputs.get("review_verdict"),
                })
            else:
                job.status = JobStatus.FAILED
                job.error_log = updated_state.error_message

            job.completed_at = datetime.utcnow()
            submission.status = (
                SubmissionStatus.REPORT_READY if job.status == JobStatus.COMPLETED
                else SubmissionStatus.FAILED
            )
            await db.commit()

        except Exception as e:
            job.status = JobStatus.FAILED
            job.error_log = str(e)
            job.completed_at = datetime.utcnow()
            await db.commit()


async def _run_review_inline(job_id: str, source_job_id: str, review_type: str):
    """Run the AI Review Agent inline against a completed analysis."""
    from src.shared.db.session import async_session_factory
    from src.medstats.agents.review_agent import ReviewAgent
    from src.agentic_core.agents.base_agent import MissionState
    from datetime import datetime
    import json

    async with async_session_factory() as db:
        result = await db.execute(select(AnalysisJob).where(AnalysisJob.id == job_id))
        job = result.scalar_one_or_none()
        if not job:
            return

        job.status = JobStatus.RUNNING
        job.started_at = datetime.utcnow()
        job.current_step = "AI Review in progress"
        await db.commit()

        try:
            # Load source analysis results
            source_result = await db.execute(
                select(AnalysisJob).where(AnalysisJob.id == source_job_id)
            )
            source_job = source_result.scalar_one_or_none()
            if not source_job or not source_job.result_data:
                job.status = JobStatus.FAILED
                job.error_log = "Source analysis job not found"
                job.completed_at = datetime.utcnow()
                await db.commit()
                return

            source_data = json.loads(source_job.result_data) if isinstance(source_job.result_data, str) else source_job.result_data

            state = MissionState(
                submission_id=str(job.submission_id),
                job_id=job_id,
            )
            state.agent_outputs.update(source_data)

            reviewer = ReviewAgent(services=None)
            state = await reviewer.run(state)

            if state.status != "FAILED":
                job.status = JobStatus.COMPLETED
                job.progress = 100
                # Merge review results with source data
                source_data["review_summary"] = state.agent_outputs.get("review_summary")
                source_data["review_verdict"] = state.agent_outputs.get("review_verdict")
                job.result_data = json.dumps(source_data)
            else:
                job.status = JobStatus.FAILED
                job.error_log = state.error_message

            job.completed_at = datetime.utcnow()
            await db.commit()

        except Exception as e:
            job.status = JobStatus.FAILED
            job.error_log = str(e)
            job.completed_at = datetime.utcnow()
            await db.commit()


async def _run_correction_inline(job_id: str, correction_data: dict):
    """Run the correction agent inline."""
    from src.shared.db.session import async_session_factory
    from src.medstats.agents.correction_agent import CorrectionAgent
    from src.agentic_core.agents.base_agent import MissionState
    from datetime import datetime
    import json

    async with async_session_factory() as db:
        result = await db.execute(select(AnalysisJob).where(AnalysisJob.id == job_id))
        job = result.scalar_one_or_none()
        if not job:
            return

        job.status = JobStatus.RUNNING
        job.started_at = datetime.utcnow()
        await db.commit()

        try:
            # Load previous result
            prev_result = await db.execute(
                select(AnalysisJob)
                .where(AnalysisJob.submission_id == job.submission_id)
                .where(AnalysisJob.status == JobStatus.COMPLETED)
                .order_by(AnalysisJob.completed_at.desc())
            )
            prev_job = prev_result.scalar_one_or_none()
            prev_data = json.loads(prev_job.result_data) if prev_job and prev_job.result_data else {}

            state = MissionState(
                submission_id=str(job.submission_id),
                job_id=job_id,
            )
            state.agent_outputs.update(prev_data)
            state.agent_outputs["professor_feedback"] = correction_data.get("feedback", "")
            state.agent_outputs["feedback_type"] = correction_data.get("feedback_type", "general_feedback")
            state.agent_outputs["target_variables"] = correction_data.get("target_variables", [])

            agent = CorrectionAgent(services=None)
            state = await agent.run(state)

            if state.status != "FAILED":
                job.status = JobStatus.COMPLETED
                job.progress = 100
                job.result_data = json.dumps({
                    "report_sections": state.agent_outputs.get("report_sections", []),
                    "charts": state.agent_outputs.get("charts", []),
                    "test_results": state.agent_outputs.get("test_results", []),
                    "title": state.agent_outputs.get("title", ""),
                    "methodology_note": state.agent_outputs.get("methodology_note", ""),
                    "summary": state.agent_outputs.get("summary", ""),
                    "report_version": state.agent_outputs.get("report_version", 2),
                    "correction_history": state.agent_outputs.get("correction_history", []),
                })
            else:
                job.status = JobStatus.FAILED
                job.error_log = state.error_message

            job.completed_at = datetime.utcnow()
            await db.commit()

        except Exception as e:
            job.status = JobStatus.FAILED
            job.error_log = str(e)
            job.completed_at = datetime.utcnow()
            await db.commit()


import os
