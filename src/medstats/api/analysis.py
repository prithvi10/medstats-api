"""
Analysis API routes.
Supports both Celery (production) and inline (local dev) pipeline execution.
"""
import json
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, HTMLResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text
from uuid import UUID
from pydantic import BaseModel
from typing import Optional, Dict
from src.shared.db.session import get_db
from src.shared.security.auth import get_current_user
from src.shared.config.settings import get_settings
from src.medstats.db.models import (
    Submission, AnalysisJob, SubmissionStatus, JobStatus, UserRole, Report,
    AgentCorrection,
)
from src.medstats.services.review_context import get_submission_for_professor

router = APIRouter()
settings = get_settings()


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def safe_json_dumps(obj):
    return json.dumps(obj, cls=NumpyEncoder)


def _agent_state_snapshot(outputs: dict) -> dict:
    """Persist fields needed for revisions and professor review context."""
    return {
        "report_sections": outputs.get("report_sections", []),
        "charts": outputs.get("charts", []),
        "test_results": outputs.get("test_results", []),
        "frequency_tables": outputs.get("frequency_tables", []),
        "title": outputs.get("title", ""),
        "intent": outputs.get("intent", ""),
        "methodology_note": outputs.get("methodology_note", ""),
        "summary": outputs.get("summary", ""),
        "variables": outputs.get("variables", []),
        "study_design": outputs.get("study_design", ""),
        "sample_size": outputs.get("sample_size", 0),
        "groups": outputs.get("groups"),
        "dataframe_json": outputs.get("dataframe_json", ""),
        "review_summary": outputs.get("review_summary"),
        "review_verdict": outputs.get("review_verdict"),
        "report_version": outputs.get("report_version", 1),
        "literature_matches": outputs.get("literature_matches", []),
        "rag_test_rationale": outputs.get("rag_test_rationale", {}),
    }


class CorrectionRequest(BaseModel):
    feedback: str
    feedback_type: str = "general_feedback"
    target_variables: list = []


class ReviewRequest(BaseModel):
    """Request to run AI review on a completed analysis."""
    review_type: str = "full"  # full, mechanical_only, methodology_only


class ExportChartRequest(BaseModel):
    """Optional client-rendered chart images (variable key -> base64 data URI)."""
    chart_images: Dict[str, str] = {}


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
        .limit(1)
    )
    job = result.scalars().first()
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
    result = await db.execute(select(Submission).where(Submission.id == submission_id))
    submission = result.scalar_one_or_none()
    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found")

    # Allow the student who owns the submission or professors/admins
    if current_user["role"] == UserRole.STUDENT.value and str(submission.student_id) != current_user["sub"]:
        raise HTTPException(status_code=403, detail="Not authorized")

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
        .limit(1)
    )
    completed_job = job_result.scalars().first()
    if not completed_job or not completed_job.agent_state:
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
    if current_user["role"] == UserRole.STUDENT.value:
        sub_result = await db.execute(select(Submission).where(Submission.id == submission_id))
        submission = sub_result.scalar_one_or_none()
        if not submission or str(submission.student_id) != current_user["sub"]:
            raise HTTPException(status_code=403, detail="Not authorized")
    elif current_user["role"] == UserRole.PROFESSOR.value:
        submission = await get_submission_for_professor(db, current_user["sub"], submission_id)
        if not submission:
            raise HTTPException(status_code=404, detail="Submission not found")
    job_result = await db.execute(
        select(AnalysisJob)
        .where(AnalysisJob.submission_id == submission_id)
        .where(AnalysisJob.status == JobStatus.COMPLETED)
        .order_by(AnalysisJob.completed_at.desc())
        .limit(1)
    )
    job = job_result.scalars().first()
    if not job or not job.agent_state:
        raise HTTPException(status_code=404, detail="No review results found")

    agent_state = json.loads(job.agent_state) if isinstance(job.agent_state, str) else job.agent_state

    review_summary = agent_state.get("review_summary")
    if not review_summary:
        raise HTTPException(status_code=404, detail="No AI review in latest completed job")

    return {
        "submission_id": str(submission_id),
        "review_summary": review_summary,
        "review_verdict": agent_state.get("review_verdict", "pending"),
    }


async def _get_latest_report_record(db: AsyncSession, submission_id: UUID) -> Report:
    """Fetch the latest report for a submission."""
    report_result = await db.execute(
        select(Report)
        .where(Report.submission_id == submission_id)
        .order_by(Report.version.desc(), Report.generated_at.desc())
        .limit(1)
    )
    report = report_result.scalars().first()
    if not report:
        raise HTTPException(status_code=404, detail="No report generated yet")
    return report


@router.get("/submissions/{submission_id}/export/preview")
async def export_report_preview(
    submission_id: UUID,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return HTML preview of the report for in-browser review before export."""
    return await _build_export_preview(submission_id, db, chart_images={})


@router.post("/submissions/{submission_id}/export/preview")
async def export_report_preview_with_charts(
    submission_id: UUID,
    body: ExportChartRequest,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return HTML preview using client-rendered chart images when provided."""
    return await _build_export_preview(submission_id, db, chart_images=body.chart_images)


async def _build_export_preview(
    submission_id: UUID,
    db: AsyncSession,
    chart_images: Dict[str, str],
):
    result = await db.execute(select(Submission).where(Submission.id == submission_id))
    submission = result.scalar_one_or_none()
    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found")

    report = await _get_latest_report_record(db, submission_id)

    from src.medstats.services.export_service import generate_preview_html, materialize_client_charts

    chart_overrides = materialize_client_charts(chart_images)
    html = generate_preview_html(
        report_sections=report.sections or [],
        charts=report.charts or [],
        title=submission.title or "Report",
        methodology_note=report.methodology_note or "",
        summary=report.summary or "",
        chart_overrides=chart_overrides,
    )
    return HTMLResponse(content=html)


@router.get("/submissions/{submission_id}/export/{format}")
async def export_report(
    submission_id: UUID,
    format: str,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Export the generated report as DOCX or PDF (legacy GET without client charts)."""
    return await _export_report_file(submission_id, format, db, chart_images={})


@router.post("/submissions/{submission_id}/export/{format}")
async def export_report_with_charts(
    submission_id: UUID,
    format: str,
    body: ExportChartRequest,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Export report using high-resolution client-rendered chart images."""
    return await _export_report_file(submission_id, format, db, chart_images=body.chart_images)


async def _export_report_file(
    submission_id: UUID,
    format: str,
    db: AsyncSession,
    chart_images: Dict[str, str],
):
    """Export the generated report as DOCX or PDF."""
    if format not in ("docx", "pdf"):
        raise HTTPException(status_code=400, detail="Format must be 'docx' or 'pdf'")

    result = await db.execute(select(Submission).where(Submission.id == submission_id))
    submission = result.scalar_one_or_none()
    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found")

    report = await _get_latest_report_record(db, submission_id)

    from src.medstats.services.export_service import (
        export_report_to_docx,
        export_report_to_pdf,
        materialize_client_charts,
        persist_export_to_storage,
    )

    chart_overrides = materialize_client_charts(chart_images)
    export_fn = export_report_to_docx if format == "docx" else export_report_to_pdf
    filepath = export_fn(
        report_sections=report.sections or [],
        charts=report.charts or [],
        title=submission.title or "Report",
        methodology_note=report.methodology_note or "",
        summary=report.summary or "",
        submission_id=str(submission_id),
        chart_overrides=chart_overrides,
    )

    export_url = persist_export_to_storage(filepath, str(submission_id), format)
    if format == "docx":
        report.docx_url = export_url
    else:
        report.pdf_url = export_url
    await db.commit()

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
    from src.medstats.agents.literature_match import LiteratureMatchAgent
    from src.agentic_core.agents.base_agent import MissionState
    from src.agentic_core.agents.orchestrator import MissionControl
    from src.agentic_core.llm.service import get_llm_service
    from datetime import datetime
    llm_service = get_llm_service()

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

            # Load file records for this submission
            from src.medstats.db.models import SubmissionFile as SF
            files_result = await db.execute(select(SF).where(SF.submission_id == job.submission_id))
            file_records = files_result.scalars().all()

            state = MissionState(
                submission_id=str(job.submission_id),
                job_id=job_id,
            )
            state.agent_outputs["title"] = submission.title or ""
            state.agent_outputs["intent"] = submission.intent_description or ""
            state.agent_outputs["files"] = [
                {"path": f.file_url, "type": f.file_type, "name": f.original_name}
                for f in file_records
            ]

            agents = [
                DocumentParserAgent(services=None),
                TestSelectorAgent(services=None),
                ReportGeneratorAgent(services=llm_service),
                LiteratureMatchAgent(services=llm_service),
            ]
            orchestrator = MissionControl(agents=agents)

            async for updated_state in orchestrator.execute(state):
                job.progress = updated_state.progress
                step_names = [
                    "Parsing documents",
                    "Running statistical tests",
                    "Generating report",
                    "Matching literature",
                ]
                idx = min(updated_state.current_agent_index, len(step_names) - 1)
                job.current_step = step_names[idx]
                await db.commit()

            # Run AI Review as post-pipeline step
            from src.medstats.agents.review_agent import ReviewAgent
            if updated_state.status == "COMPLETED":
                job.current_step = "AI Review"
                await db.commit()
                reviewer = ReviewAgent(services=llm_service)
                updated_state = await reviewer.run(updated_state)

            # Use raw SQL for the final status writes — ORM dirty-tracking is
            # unreliable after many mid-loop commits on the same session.
            now = datetime.utcnow()
            has_ai_review = bool(updated_state.agent_outputs.get("review_summary"))

            if updated_state.status == "COMPLETED":
                final_job_status = "COMPLETED"
                agent_state_json = safe_json_dumps(_agent_state_snapshot(updated_state.agent_outputs))
                sub_status = "AI_REVIEWED" if has_ai_review else "REPORT_READY"

                # Insert report via ORM (new object, no tracking issues)
                report = Report(
                    submission_id=job.submission_id,
                    analysis_job_id=job.id,
                    version=1,
                    sections=updated_state.agent_outputs.get("report_sections", []),
                    charts=updated_state.agent_outputs.get("charts", []),
                    summary=updated_state.agent_outputs.get("summary", ""),
                    methodology_note=updated_state.agent_outputs.get("methodology_note", ""),
                )
                db.add(report)
                await db.flush()  # write report first; fail early if sections have bad data
            else:
                final_job_status = "FAILED"
                agent_state_json = None
                sub_status = "FAILED"

            await db.execute(
                text(
                    "UPDATE analysis_jobs SET status=:status, progress=100, "
                    "current_step='Analysis complete', agent_state=:agent_state, "
                    "completed_at=:now WHERE id=:job_id"
                ),
                {
                    "status": final_job_status,
                    "agent_state": agent_state_json,
                    "now": now,
                    "job_id": job_id,
                },
            )
            await db.execute(
                text("UPDATE submissions SET status=:status, updated_at=:now WHERE id=:sid"),
                {"status": sub_status, "now": now, "sid": str(job.submission_id)},
            )
            await db.commit()

        except Exception as e:
            logger.error("Pipeline failed", job_id=job_id, error=str(e))
            try:
                await db.rollback()
                await db.execute(
                    text(
                        "UPDATE analysis_jobs SET status='FAILED', error_log=:err, "
                        "current_step='Failed', completed_at=:now WHERE id=:job_id"
                    ),
                    {"err": str(e)[:2000], "now": datetime.utcnow(), "job_id": job_id},
                )
                await db.execute(
                    text(
                        "UPDATE submissions SET status='FAILED', updated_at=:now WHERE id=:sid"
                    ),
                    {"now": datetime.utcnow(), "sid": str(job.submission_id)},
                )
                await db.commit()
            except Exception as e2:
                logger.error("Failed to persist pipeline error", job_id=job_id, error=str(e2))
            await db.commit()


async def _run_review_inline(job_id: str, source_job_id: str, review_type: str):
    """Run the AI Review Agent inline against a completed analysis."""
    from src.shared.db.session import async_session_factory
    from src.medstats.agents.review_agent import ReviewAgent
    from src.agentic_core.agents.base_agent import MissionState
    from src.agentic_core.llm.service import get_llm_service
    from datetime import datetime
    llm_service = get_llm_service()

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
            if not source_job or not source_job.agent_state:
                job.status = JobStatus.FAILED
                job.error_log = "Source analysis job not found"
                job.completed_at = datetime.utcnow()
                await db.commit()
                return

            source_data = json.loads(source_job.agent_state) if isinstance(source_job.agent_state, str) else source_job.agent_state

            state = MissionState(
                submission_id=str(job.submission_id),
                job_id=job_id,
            )
            state.agent_outputs.update(source_data)

            reviewer = ReviewAgent(services=llm_service)
            state = await reviewer.run(state)

            if state.status != "FAILED":
                job.status = JobStatus.COMPLETED
                job.progress = 100
                # Merge review results with source data
                source_data["review_summary"] = state.agent_outputs.get("review_summary")
                source_data["review_verdict"] = state.agent_outputs.get("review_verdict")
                job.agent_state = safe_json_dumps(source_data)

                sub_result = await db.execute(
                    select(Submission).where(Submission.id == job.submission_id)
                )
                submission = sub_result.scalar_one_or_none()
                if submission and submission.status in (
                    SubmissionStatus.REPORT_READY,
                    SubmissionStatus.ANALYZING,
                ):
                    submission.status = SubmissionStatus.AI_REVIEWED
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
    from src.agentic_core.llm.service import get_llm_service
    from datetime import datetime
    llm_service = get_llm_service()

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
            prev_data = json.loads(prev_job.agent_state) if prev_job and prev_job.agent_state else {}

            state = MissionState(
                submission_id=str(job.submission_id),
                job_id=job_id,
            )
            state.agent_outputs.update(prev_data)
            state.agent_outputs["professor_feedback"] = correction_data.get("feedback", "")
            state.agent_outputs["feedback_type"] = correction_data.get("feedback_type", "general_feedback")
            state.agent_outputs["target_variables"] = correction_data.get("target_variables", [])

            agent = CorrectionAgent(services=llm_service)
            state = await agent.run(state)

            if state.status != "FAILED":
                job.status = JobStatus.COMPLETED
                job.progress = 100
                job.agent_state = safe_json_dumps(_agent_state_snapshot(state.agent_outputs))
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


async def _run_revision_inline(job_id: str, revision_payload: dict):
    """Apply structured professor review via RevisionResolverAgent."""
    from src.shared.db.session import async_session_factory
    from src.medstats.agents.revision_resolver import RevisionResolverAgent
    from src.agentic_core.agents.base_agent import MissionState
    from src.agentic_core.llm.service import get_llm_service
    from src.medstats.services.review_context import get_latest_analysis_state, get_latest_report
    from datetime import datetime

    llm_service = get_llm_service()

    async with async_session_factory() as db:
        result = await db.execute(select(AnalysisJob).where(AnalysisJob.id == job_id))
        job = result.scalar_one_or_none()
        if not job:
            return

        job.status = JobStatus.RUNNING
        job.started_at = datetime.utcnow()
        job.current_step = "Resolving professor feedback"
        await db.commit()

        try:
            prev_data = await get_latest_analysis_state(db, job.submission_id)
            if not prev_data:
                raise ValueError("No prior analysis state found for revision")

            sub_result = await db.execute(select(Submission).where(Submission.id == job.submission_id))
            submission = sub_result.scalar_one_or_none()

            latest_report = await get_latest_report(db, job.submission_id)
            current_version = latest_report.version if latest_report else 1

            state = MissionState(
                submission_id=str(job.submission_id),
                job_id=job_id,
            )
            state.agent_outputs.update(prev_data)
            state.agent_outputs["report_version"] = current_version
            state.agent_outputs["structured_review"] = revision_payload

            agent = RevisionResolverAgent(services=llm_service)
            state = await agent.run(state)

            review_id = revision_payload.get("review_id")

            if state.status != "FAILED":
                job.status = JobStatus.COMPLETED
                job.progress = 100
                job.agent_state = safe_json_dumps(_agent_state_snapshot(state.agent_outputs))

                new_report = Report(
                    submission_id=job.submission_id,
                    analysis_job_id=job.id,
                    version=current_version + 1,
                    sections=state.agent_outputs.get("report_sections", []),
                    charts=state.agent_outputs.get("charts", []),
                    summary=state.agent_outputs.get("summary", ""),
                    methodology_note=state.agent_outputs.get("methodology_note", ""),
                )
                db.add(new_report)
                await db.flush()

                if review_id:
                    correction = AgentCorrection(
                        submission_id=job.submission_id,
                        review_id=review_id,
                        correction_prompt=json.dumps(revision_payload),
                        changes_made=state.agent_outputs.get("revision_history", [])[-1] if state.agent_outputs.get("revision_history") else {},
                        new_report_id=new_report.id,
                    )
                    db.add(correction)

                if submission:
                    submission.status = SubmissionStatus.UNDER_REVIEW
            else:
                job.status = JobStatus.FAILED
                job.error_log = state.error_message
                if submission:
                    submission.status = SubmissionStatus.REVISION

            job.completed_at = datetime.utcnow()
            await db.commit()

        except Exception as e:
            job.status = JobStatus.FAILED
            job.error_log = str(e)
            job.completed_at = datetime.utcnow()
            sub_result = await db.execute(select(Submission).where(Submission.id == job.submission_id))
            submission = sub_result.scalar_one_or_none()
            if submission:
                submission.status = SubmissionStatus.REVISION
            await db.commit()


import os
