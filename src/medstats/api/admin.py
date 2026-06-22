from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from src.shared.db.session import get_db
from src.shared.security.auth import get_current_user
from src.medstats.db.models import User, Submission, AnalysisJob, UserRole

router = APIRouter()


def _require_admin(current_user: dict):
    if current_user.get("role") != UserRole.ADMIN.value:
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user


@router.get("/dashboard")
async def admin_dashboard(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    _require_admin(current_user)

    total_users = await db.execute(select(func.count(User.id)))
    total_submissions = await db.execute(select(func.count(Submission.id)))
    total_jobs = await db.execute(select(func.count(AnalysisJob.id)))

    students = await db.execute(
        select(func.count(User.id)).where(User.role == UserRole.STUDENT)
    )
    professors = await db.execute(
        select(func.count(User.id)).where(User.role == UserRole.PROFESSOR)
    )

    return {
        "total_users": total_users.scalar(),
        "total_submissions": total_submissions.scalar(),
        "total_jobs": total_jobs.scalar(),
        "students_count": students.scalar(),
        "professors_count": professors.scalar(),
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
                "is_active": u.is_active,
                "created_at": u.created_at.isoformat(),
            }
            for u in users
        ]
    }
