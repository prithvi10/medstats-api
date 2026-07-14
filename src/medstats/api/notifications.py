from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, update
from pydantic import BaseModel
from typing import List, Optional
from uuid import UUID

from src.shared.db.session import get_db
from src.shared.security.auth import get_current_user
from src.medstats.db.models import Notification

router = APIRouter()


class NotificationResponse(BaseModel):
    id: str
    type: str
    title: str
    message: Optional[str]
    submission_id: Optional[str]
    is_read: bool
    created_at: str


class NotificationListResponse(BaseModel):
    notifications: List[NotificationResponse]
    total: int


class UnreadCountResponse(BaseModel):
    count: int


@router.get("", response_model=NotificationListResponse)
async def list_notifications(
    limit: int = 50,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    user_id = current_user["sub"]
    result = await db.execute(
        select(Notification)
        .where(Notification.user_id == user_id)
        .order_by(Notification.created_at.desc())
        .limit(limit)
    )
    notifications = result.scalars().all()
    return NotificationListResponse(
        notifications=[
            NotificationResponse(
                id=str(n.id),
                type=n.type,
                title=n.title,
                message=n.message,
                submission_id=str(n.submission_id) if n.submission_id else None,
                is_read=n.is_read,
                created_at=n.created_at.isoformat(),
            )
            for n in notifications
        ],
        total=len(notifications),
    )


@router.get("/unread-count", response_model=UnreadCountResponse)
async def unread_count(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    user_id = current_user["sub"]
    result = await db.execute(
        select(func.count())
        .select_from(Notification)
        .where(Notification.user_id == user_id, Notification.is_read == False)  # noqa: E712
    )
    return UnreadCountResponse(count=result.scalar() or 0)


@router.patch("/{notification_id}/read")
async def mark_notification_read(
    notification_id: UUID,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Notification).where(
            Notification.id == notification_id,
            Notification.user_id == current_user["sub"],
        )
    )
    notification = result.scalar_one_or_none()
    if not notification:
        raise HTTPException(status_code=404, detail="Notification not found")

    notification.is_read = True
    await db.commit()
    return {"message": "Notification marked as read"}


@router.patch("/read-all")
async def mark_all_read(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    await db.execute(
        update(Notification)
        .where(
            Notification.user_id == current_user["sub"],
            Notification.is_read == False,  # noqa: E712
        )
        .values(is_read=True)
    )
    await db.commit()
    return {"message": "All notifications marked as read"}
