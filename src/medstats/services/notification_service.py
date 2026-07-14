"""Notification creation and email dispatch."""

from __future__ import annotations

import uuid
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.medstats.db.models import Notification, User
from src.medstats.services import email_service
from src.medstats.services.email_templates import (
    review_completed_email,
    review_requested_email,
    review_started_email,
    submission_approved_email,
)
from src.shared.logging.logger import get_logger

logger = get_logger(__name__)

EMAIL_TEMPLATE_BY_TYPE = {
    "review_requested": review_requested_email,
    "review_started": review_started_email,
    "review_completed": review_completed_email,
    "approved": submission_approved_email,
}


async def create_notification(
    db: AsyncSession,
    user_id: str,
    type: str,
    title: str,
    message: str,
    submission_id: Optional[str] = None,
    send_email: bool = True,
    email_context: Optional[dict] = None,
) -> Notification:
    """Create an in-app notification and optionally send an email."""
    notification = Notification(
        id=uuid.uuid4(),
        user_id=user_id,
        type=type,
        title=title,
        message=message,
        submission_id=submission_id,
        is_read=False,
    )
    db.add(notification)
    await db.flush()

    if send_email:
        await _dispatch_email(db, user_id, type, email_context or {})

    return notification


async def _dispatch_email(
    db: AsyncSession,
    user_id: str,
    notification_type: str,
    context: dict,
) -> None:
    template_fn = EMAIL_TEMPLATE_BY_TYPE.get(notification_type)
    if not template_fn:
        return

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user or not user.email:
        return

    try:
        subject, html_body, text_body = template_fn(**context)
        await email_service.send_email(
            to=user.email,
            subject=subject,
            html_body=html_body,
            text_body=text_body,
        )
    except Exception as exc:
        logger.error(
            "[notification] email failed user_id=%s type=%s error=%s",
            user_id,
            notification_type,
            exc,
        )


async def notify_review_requested(
    db: AsyncSession,
    professor_id: str,
    student_name: str,
    submission_id: str,
    submission_title: str,
    professor_name: str,
) -> None:
    await create_notification(
        db=db,
        user_id=professor_id,
        type="review_requested",
        title="New review request",
        message=f"{student_name} submitted \"{submission_title}\" for your review.",
        submission_id=submission_id,
        email_context={
            "professor_name": professor_name,
            "student_name": student_name,
            "submission_title": submission_title,
        },
    )


async def notify_review_started(
    db: AsyncSession,
    student_id: str,
    professor_name: str,
    submission_id: str,
    submission_title: str,
    student_name: str,
) -> None:
    await create_notification(
        db=db,
        user_id=student_id,
        type="review_started",
        title="Professor started review",
        message=f"Professor {professor_name} started reviewing \"{submission_title}\".",
        submission_id=submission_id,
        email_context={
            "student_name": student_name,
            "professor_name": professor_name,
            "submission_title": submission_title,
        },
    )


async def notify_review_completed(
    db: AsyncSession,
    student_id: str,
    professor_name: str,
    submission_id: str,
    submission_title: str,
    student_name: str,
    decision: str,
) -> None:
    await create_notification(
        db=db,
        user_id=student_id,
        type="review_completed",
        title="Professor review completed",
        message=(
            f"Professor {professor_name} completed the review of "
            f"\"{submission_title}\" ({decision.replace('_', ' ')})."
        ),
        submission_id=submission_id,
        email_context={
            "student_name": student_name,
            "professor_name": professor_name,
            "submission_title": submission_title,
            "decision": decision,
        },
    )


async def notify_submission_approved(
    db: AsyncSession,
    student_id: str,
    professor_name: str,
    submission_id: str,
    submission_title: str,
    student_name: str,
) -> None:
    await create_notification(
        db=db,
        user_id=student_id,
        type="approved",
        title="Analysis approved",
        message=f"Professor {professor_name} approved \"{submission_title}\".",
        submission_id=submission_id,
        email_context={
            "student_name": student_name,
            "professor_name": professor_name,
            "submission_title": submission_title,
        },
    )
