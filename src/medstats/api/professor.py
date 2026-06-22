from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel
from typing import Optional, List
from uuid import UUID
from src.shared.db.session import get_db
from src.shared.security.auth import get_current_user
from src.medstats.db.models import (
    User, Submission, Report, ProfessorReview, ProfessorStudent,
    ReviewDecision, SubmissionStatus, UserRole
)

router = APIRouter()


class ReviewRequest(BaseModel):
    decision: ReviewDecision
    feedback: Optional[str] = None
    section_comments: Optional[dict] = None


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
    result = await db.execute(
        select(Submission)
        .where(Submission.professor_id == current_user["sub"])
        .order_by(Submission.created_at.desc())
    )
    submissions = result.scalars().all()

    items = []
    for s in submissions:
        student_result = await db.execute(select(User).where(User.id == s.student_id))
        student = student_result.scalar_one_or_none()
        items.append({
            "id": str(s.id),
            "title": s.title,
            "status": s.status.value,
            "student_name": student.name if student else "Unknown",
            "student_email": student.email if student else "",
            "created_at": s.created_at.isoformat(),
        })
    return {"submissions": items, "total": len(items)}


@router.get("/submissions/{submission_id}/report")
async def get_report_for_review(
    submission_id: UUID,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    _require_professor(current_user)
    result = await db.execute(
        select(Submission).where(
            Submission.id == submission_id,
            Submission.professor_id == current_user["sub"],
        )
    )
    submission = result.scalar_one_or_none()
    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found")

    report_result = await db.execute(
        select(Report)
        .where(Report.submission_id == submission_id)
        .order_by(Report.version.desc())
    )
    report = report_result.scalar_one_or_none()
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
    request: ReviewRequest,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    _require_professor(current_user)

    sub_result = await db.execute(
        select(Submission).where(
            Submission.id == submission_id,
            Submission.professor_id == current_user["sub"],
        )
    )
    submission = sub_result.scalar_one_or_none()
    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found")

    report_result = await db.execute(
        select(Report)
        .where(Report.submission_id == submission_id)
        .order_by(Report.version.desc())
    )
    report = report_result.scalar_one_or_none()
    if not report:
        raise HTTPException(status_code=404, detail="No report to review")

    review = ProfessorReview(
        report_id=report.id,
        professor_id=current_user["sub"],
        decision=request.decision,
        feedback=request.feedback,
        section_comments=request.section_comments,
    )
    db.add(review)

    if request.decision == ReviewDecision.APPROVED:
        submission.status = SubmissionStatus.APPROVED
    elif request.decision == ReviewDecision.NEEDS_REVISION:
        submission.status = SubmissionStatus.REVISION
    elif request.decision == ReviewDecision.REJECTED:
        submission.status = SubmissionStatus.SUBMITTED

    await db.commit()
    return {"message": "Review submitted", "decision": request.decision.value}
