#!/usr/bin/env python3
"""Send stack failure alerts via email and webhook.

Usage:
  python scripts/alert_notify.py "API health check failed"
  python scripts/alert_notify.py --subject "Worker errors" --body "Traceback in celery..."
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

for _env in (".env.beta", ".env"):
    if os.path.exists(_env):
        load_dotenv(_env)
        break


async def main() -> int:
    parser = argparse.ArgumentParser(description="Send MedStats stack alert")
    parser.add_argument("message", nargs="?", help="Alert message body")
    parser.add_argument("--subject", default="Stack failure detected")
    parser.add_argument("--body", help="Override message body")
    args = parser.parse_args()

    body = args.body or args.message
    if not body:
        parser.error("Provide a message or --body")

    from src.medstats.services import email_service

    await email_service.send_alert(args.subject, body)
    print(f"Alert sent: {args.subject}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
