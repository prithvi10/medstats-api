"""
SES Email Service — sends transactional emails via Amazon SES.

Required env vars:
    AWS_REGION
    AWS_ACCESS_KEY_ID
    AWS_SECRET_ACCESS_KEY
    SES_FROM_ADDRESS
"""

from __future__ import annotations

import asyncio
import os
from typing import Optional

from src.shared.logging.logger import get_logger

logger = get_logger(__name__)

_ses_client = None


def _get_ses_client():
    """Lazily create a boto3 SES client."""
    global _ses_client
    if _ses_client is None:
        try:
            import boto3
        except ImportError as exc:
            raise ImportError(
                "boto3 is required for SES email. Install it with: pip install boto3"
            ) from exc

        _ses_client = boto3.client(
            "ses",
            region_name=os.getenv("AWS_REGION", "us-east-1"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        )
    return _ses_client


async def send_email(
    to: str,
    subject: str,
    html_body: str,
    text_body: str,
    reply_to: Optional[str] = None,
) -> dict:
    """Send a single email via SES."""
    from_address = os.getenv("SES_FROM_ADDRESS", '"MedStats AI" <hello@medstats.ai>')
    client = _get_ses_client()

    message: dict = {
        "Subject": {"Data": subject, "Charset": "UTF-8"},
        "Body": {
            "Html": {"Data": html_body, "Charset": "UTF-8"},
            "Text": {"Data": text_body, "Charset": "UTF-8"},
        },
    }

    kwargs: dict = {
        "Source": from_address,
        "Destination": {"ToAddresses": [to]},
        "Message": message,
    }
    if reply_to:
        kwargs["ReplyToAddresses"] = [reply_to]

    loop = asyncio.get_running_loop()
    response = await loop.run_in_executor(None, lambda: client.send_email(**kwargs))

    message_id = response.get("MessageId", "")
    logger.info("[SES] sent to=%s subject=%r message_id=%s", to, subject, message_id)
    return {"message_id": message_id}
