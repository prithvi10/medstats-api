from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel
from typing import Optional, List
from uuid import UUID
from src.shared.db.session import get_db
from src.shared.security.auth import get_current_user
from src.medstats.db.models import (
    User, Submission, SubmissionFile, SubmissionStatus, Report, UserRole
)

router = APIRouter()


class SubmissionResponse(BaseModel):
    id: str
    title: str
    status: SubmissionStatus
    intent_description: Optional[str]
    professor_id: Optional[str]
    created_at: str
    updated_at: str
    files: List[dict] = []

    class Config:
        from_attributes = True


class SubmissionListResponse(BaseModel):
    submissions: List[SubmissionResponse]
    total: int


def _require_student(current_user: dict):
    if current_user.get("role") != UserRole.STUDENT.value:
        raise HTTPException(status_code=403, detail="Student access required")
    return current_user


@router.get("/submissions", response_model=SubmissionListResponse)
async def list_submissions(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    _require_student(current_user)
    result = await db.execute(
        select(Submission)
        .where(Submission.student_id == current_user["sub"])
        .order_by(Submission.created_at.desc())
    )
    submissions = result.scalars().all()

    items = []
    for s in submissions:
        await db.refresh(s, ["files"])
        items.append(SubmissionResponse(
            id=str(s.id),
            title=s.title,
            status=s.status,
            intent_description=s.intent_description,
            professor_id=str(s.professor_id) if s.professor_id else None,
            created_at=s.created_at.isoformat(),
            updated_at=s.updated_at.isoformat(),
            files=[{"id": str(f.id), "name": f.original_name, "type": f.file_type} for f in s.files],
        ))
    return SubmissionListResponse(submissions=items, total=len(items))


@router.post("/submissions", response_model=SubmissionResponse, status_code=201)
async def create_submission(
    title: str = Form(...),
    intent_description: Optional[str] = Form(None),
    professor_id: Optional[str] = Form(None),
    files: List[UploadFile] = File(...),
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    _require_student(current_user)

    submission = Submission(
        student_id=current_user["sub"],
        title=title,
        intent_description=intent_description,
        professor_id=professor_id if professor_id else None,
        status=SubmissionStatus.SUBMITTED,
    )
    db.add(submission)
    await db.flush()

    saved_files = []
    for uploaded_file in files:
        file_type = _classify_file(uploaded_file.filename)
        content = await uploaded_file.read()

        # For MVP, store locally. Later: upload to S3/R2
        file_path = f"uploads/{submission.id}/{uploaded_file.filename}"
        import os
        os.makedirs(f"uploads/{submission.id}", exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(content)

        sub_file = SubmissionFile(
            submission_id=submission.id,
            file_url=file_path,
            file_type=file_type,
            original_name=uploaded_file.filename,
            file_size=len(content),
        )
        db.add(sub_file)
        saved_files.append(sub_file)

    await db.commit()
    await db.refresh(submission)

    return SubmissionResponse(
        id=str(submission.id),
        title=submission.title,
        status=submission.status,
        intent_description=submission.intent_description,
        professor_id=str(submission.professor_id) if submission.professor_id else None,
        created_at=submission.created_at.isoformat(),
        updated_at=submission.updated_at.isoformat(),
        files=[{"id": str(f.id), "name": f.original_name, "type": f.file_type} for f in saved_files],
    )


@router.get("/submissions/{submission_id}", response_model=SubmissionResponse)
async def get_submission(
    submission_id: UUID,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    _require_student(current_user)
    result = await db.execute(
        select(Submission).where(
            Submission.id == submission_id,
            Submission.student_id == current_user["sub"],
        )
    )
    submission = result.scalar_one_or_none()
    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found")

    await db.refresh(submission, ["files"])
    return SubmissionResponse(
        id=str(submission.id),
        title=submission.title,
        status=submission.status,
        intent_description=submission.intent_description,
        professor_id=str(submission.professor_id) if submission.professor_id else None,
        created_at=submission.created_at.isoformat(),
        updated_at=submission.updated_at.isoformat(),
        files=[{"id": str(f.id), "name": f.original_name, "type": f.file_type} for f in submission.files],
    )


@router.get("/submissions/{submission_id}/report")
async def get_submission_report(
    submission_id: UUID,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    _require_student(current_user)
    result = await db.execute(
        select(Submission).where(
            Submission.id == submission_id,
            Submission.student_id == current_user["sub"],
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
        raise HTTPException(status_code=404, detail="No report generated yet")

    return {
        "id": str(report.id),
        "version": report.version,
        "sections": report.sections,
        "charts": report.charts,
        "summary": report.summary,
        "methodology_note": report.methodology_note,
        "pdf_url": report.pdf_url,
        "docx_url": report.docx_url,
        "generated_at": report.generated_at.isoformat(),
    }


def _classify_file(filename: str) -> str:
    lower = filename.lower()
    if lower.endswith((".xlsx", ".xls", ".csv")):
        return "data_excel" if not lower.endswith(".csv") else "data_csv"
    elif lower.endswith((".docx", ".doc", ".pdf")):
        if any(kw in lower for kw in ["master", "chart", "data", "record"]):
            return "data_excel"
        elif any(kw in lower for kw in ["discussion", "intent", "background", "protocol"]):
            return "supporting_doc"
        else:
            return "supporting_doc"
    return "supporting_doc"
