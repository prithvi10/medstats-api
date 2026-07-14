from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel
from typing import Optional, List
from uuid import UUID
from src.shared.db.session import get_db
from src.shared.security.auth import get_current_user
from src.shared.config.settings import get_settings
from src.medstats.db.models import (
    User, Submission, Report, ProfessorReview, ProfessorStudent,
    ReviewDecision, SubmissionStatus, UserRole, AnalysisJob, JobStatus,
)
from src.medstats.services.review_context import (
    get_submission_for_professor,
    list_professor_submissions,
    get_latest_analysis_state,
    build_data_profile,
    build_review_history,
    build_master_chart_preview,
    build_tests_selected,
    count_review_rounds,
    get_latest_report,
)
from src.medstats.services.notification_service import (
    notify_review_started,
    notify_review_completed,
    notify_submission_approved,
)

router = APIRouter()
settings = get_settings()


class TestOverride(BaseModel):
    variable: str
    new_test: str


class SectionFeedback(BaseModel):
    target: str
    preset_codes: List[str] = []


class StructuredReviewRequest(BaseModel):
    decision: ReviewDecision
    test_overrides: List[TestOverride] = []
    section_feedback: List[SectionFeedback] = []
    final_comment: Optional[str] = None


class StudentInfo(BaseModel):
    id: str
    name: str
    email: str
    department: Optional[str]
    submissions_count: int = 0


def _require_professor(current_user: dict):
    if current_user.get("role") != UserRole.PROFESSOR.value:
        raise HTTPException(status_code=403, detail="Professor access required")
    return current_user


def _needs_revision_agent(request: StructuredReviewRequest) -> bool:
    if request.decision != ReviewDecision.NEEDS_REVISION:
        return False
    has_overrides = any(o.new_test for o in request.test_overrides)
    has_feedback = any(f.preset_codes for f in request.section_feedback)
    has_comment = bool(request.final_comment and request.final_comment.strip())
    return has_overrides or has_feedback or has_comment


@router.get("/students")
async def list_students(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    _require_professor(current_user)
    result = await db.execute(
        select(ProfessorStudent).where(
            ProfessorStudent.professor_id == current_user["sub"]
        )
    )
    links = result.scalars().all()

    students = []
    for link in links:
        student_result = await db.execute(select(User).where(User.id == link.student_id))
        student = student_result.scalar_one_or_none()
        if student:
            sub_count = await db.execute(
                select(Submission).where(Submission.student_id == student.id)
            )
            count = len(sub_count.scalars().all())
            students.append(StudentInfo(
                id=str(student.id),
                name=student.name,
                email=student.email,
                department=student.department,
                submissions_count=count,
            ))
    return {"students": students}


@router.get("/submissions")
async def list_all_student_submissions(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    _require_professor(current_user)
    submissions = await list_professor_submissions(db, current_user["sub"])

    items = []
    for s in submissions:
        student_result = await db.execute(select(User).where(User.id == s.student_id))
        student = student_result.scalar_one_or_none()
        review_round = await count_review_rounds(db, s.id)
        items.append({
            "id": str(s.id),
            "title": s.title,
            "status": s.status.value,
            "student_name": student.name if student else "Unknown",
            "student_email": student.email if student else "",
            "review_round": review_round,
            "created_at": s.created_at.isoformat(),
            "updated_at": s.updated_at.isoformat(),
        })
    return {"submissions": items, "total": len(items)}


@router.get("/submissions/{submission_id}")
async def get_review_context(
    submission_id: UUID,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Full review context: inputs, data profile, report, AI review, history."""
    _require_professor(current_user)
    submission = await get_submission_for_professor(db, current_user["sub"], submission_id)
    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found")

    if submission.status == SubmissionStatus.UNDER_REVIEW:
        submission.status = SubmissionStatus.REVIEW_IN_PROGRESS
        student_result = await db.execute(select(User).where(User.id == submission.student_id))
        student = student_result.scalar_one_or_none()
        professor_result = await db.execute(select(User).where(User.id == current_user["sub"]))
        professor = professor_result.scalar_one_or_none()
        if student and professor:
            await notify_review_started(
                db=db,
                student_id=str(student.id),
                professor_name=professor.name,
                submission_id=str(submission.id),
                submission_title=submission.title,
                student_name=student.name,
            )
        await db.commit()

    await db.refresh(submission, ["files"])
    student_result = await db.execute(select(User).where(User.id == submission.student_id))
    student = student_result.scalar_one_or_none()

    report = await get_latest_report(db, submission_id)

    agent_state = await get_latest_analysis_state(db, submission_id)
    data_profile = build_data_profile(agent_state) if agent_state else None
    master_chart_preview = (
        build_master_chart_preview(agent_state or {}, submission.files)
        if submission.files
        else None
    )
    tests_selected = build_tests_selected(data_profile)

    ai_review = None
    if agent_state:
        review_summary = agent_state.get("review_summary")
        if review_summary:
            ai_review = {
                "review_summary": review_summary,
                "review_verdict": agent_state.get("review_verdict", "pending"),
            }

    review_history = await build_review_history(db, submission_id)
    review_round = len(review_history) + 1

    return {
        "submission": {
            "id": str(submission.id),
            "title": submission.title,
            "status": submission.status.value,
            "intent_description": submission.intent_description,
            "created_at": submission.created_at.isoformat(),
            "updated_at": submission.updated_at.isoformat(),
            "files": [
                {"id": str(f.id), "name": f.original_name, "type": f.file_type}
                for f in submission.files
            ],
        },
        "student": {
            "id": str(student.id) if student else None,
            "name": student.name if student else "Unknown",
            "email": student.email if student else "",
            "department": student.department if student else None,
        },
        "data_profile": data_profile,
        "master_chart_preview": master_chart_preview,
        "tests_selected": tests_selected,
        "report": {
            "id": str(report.id),
            "version": report.version,
            "sections": report.sections,
            "charts": report.charts,
            "summary": report.summary,
            "methodology_note": report.methodology_note,
            "generated_at": report.generated_at.isoformat(),
        } if report else None,
        "ai_review": ai_review,
        "review_history": review_history,
        "review_round": review_round,
    }


@router.get("/submissions/{submission_id}/report")
async def get_report_for_review(
    submission_id: UUID,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    _require_professor(current_user)
    submission = await get_submission_for_professor(db, current_user["sub"], submission_id)
    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found")

    report = await get_latest_report(db, submission_id)
    if not report:
        raise HTTPException(status_code=404, detail="No report available")

    return {
        "id": str(report.id),
        "version": report.version,
        "sections": report.sections,
        "charts": report.charts,
        "summary": report.summary,
        "methodology_note": report.methodology_note,
        "submission_title": submission.title,
        "generated_at": report.generated_at.isoformat(),
    }


@router.post("/submissions/{submission_id}/review")
async def submit_review(
    submission_id: UUID,
    request: StructuredReviewRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    _require_professor(current_user)

    submission = await get_submission_for_professor(db, current_user["sub"], submission_id)
    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found")

    report = await get_latest_report(db, submission_id)
    if not report:
        raise HTTPException(status_code=404, detail="No report to review")

    section_comments = {
        "test_overrides": [o.model_dump() for o in request.test_overrides if o.new_test],
        "section_feedback": [f.model_dump() for f in request.section_feedback if f.preset_codes],
        "structured": True,
    }

    review = ProfessorReview(
        report_id=report.id,
        professor_id=current_user["sub"],
        decision=request.decision,
        feedback=request.final_comment,
        section_comments=section_comments,
    )
    db.add(review)
    await db.flush()

    revision_job_id = None
    student_result = await db.execute(select(User).where(User.id == submission.student_id))
    student = student_result.scalar_one_or_none()
    professor_result = await db.execute(select(User).where(User.id == current_user["sub"]))
    professor = professor_result.scalar_one_or_none()

    if request.decision == ReviewDecision.APPROVED:
        submission.status = SubmissionStatus.APPROVED
        if student and professor:
            await notify_submission_approved(
                db=db,
                student_id=str(student.id),
                professor_name=professor.name,
                submission_id=str(submission.id),
                submission_title=submission.title,
                student_name=student.name,
            )
    elif request.decision == ReviewDecision.NEEDS_REVISION:
        submission.status = SubmissionStatus.REVIEW_COMPLETED
        if student and professor:
            await notify_review_completed(
                db=db,
                student_id=str(student.id),
                professor_name=professor.name,
                submission_id=str(submission.id),
                submission_title=submission.title,
                student_name=student.name,
                decision=request.decision.value,
            )
        if _needs_revision_agent(request):
            submission.status = SubmissionStatus.ANALYZING
            revision_job = AnalysisJob(
                submission_id=submission_id,
                status=JobStatus.QUEUED,
                current_step="Applying professor feedback",
            )
            db.add(revision_job)
            await db.flush()
            revision_job_id = str(revision_job.id)
            revision_payload = {
                "review_id": str(review.id),
                "test_overrides": [o.model_dump() for o in request.test_overrides if o.new_test],
                "section_feedback": [f.model_dump() for f in request.section_feedback if f.preset_codes],
                "final_comment": request.final_comment or "",
            }
            if settings.ENV == "production":
                from src.medstats.services.worker import run_revision_pipeline
                run_revision_pipeline.delay(revision_job_id, revision_payload)
            else:
                from src.medstats.api.analysis import _run_revision_inline
                background_tasks.add_task(_run_revision_inline, revision_job_id, revision_payload)
        else:
            submission.status = SubmissionStatus.REVISION
    elif request.decision == ReviewDecision.REJECTED:
        submission.status = SubmissionStatus.SUBMITTED

    await db.commit()

    return {
        "message": "Review submitted",
        "decision": request.decision.value,
        "review_id": str(review.id),
        "revision_job_id": revision_job_id,
        "revision_queued": revision_job_id is not None,
    }


@router.get("/submissions/{submission_id}/jobs/{job_id}")
async def get_revision_job_status(
    submission_id: UUID,
    job_id: UUID,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Poll a specific revision job (professor-only)."""
    _require_professor(current_user)
    submission = await get_submission_for_professor(db, current_user["sub"], submission_id)
    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found")

    result = await db.execute(
        select(AnalysisJob)
        .where(AnalysisJob.id == job_id)
        .where(AnalysisJob.submission_id == submission_id)
    )
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return {
        "job_id": str(job.id),
        "status": job.status.value,
        "progress": job.progress,
        "current_step": job.current_step,
        "error_log": job.error_log,
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
    }
