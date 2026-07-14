"""Email templates for MedStats notification events."""

from __future__ import annotations


def _wrap_html(title: str, body: str) -> str:
    return f"""<!DOCTYPE html>
<html>
<body style="font-family: Arial, sans-serif; line-height: 1.5; color: #111827;">
  <h2 style="color: #2563eb;">{title}</h2>
  {body}
  <p style="color: #6b7280; font-size: 12px; margin-top: 24px;">
    MedStats AI — Statistical analysis for medical thesis
  </p>
</body>
</html>"""


def review_requested_email(
    professor_name: str,
    student_name: str,
    submission_title: str,
) -> tuple[str, str, str]:
    subject = f"Review requested: {submission_title}"
    text_body = (
        f"Hello {professor_name},\n\n"
        f"{student_name} has submitted \"{submission_title}\" for your review.\n\n"
        "Please log in to the professor portal to begin the review."
    )
    html_body = _wrap_html(
        "New review request",
        f"<p>Hello {professor_name},</p>"
        f"<p><strong>{student_name}</strong> has submitted "
        f"<strong>{submission_title}</strong> for your review.</p>"
        "<p>Please log in to the professor portal to begin the review.</p>",
    )
    return subject, html_body, text_body


def review_started_email(
    student_name: str,
    professor_name: str,
    submission_title: str,
) -> tuple[str, str, str]:
    subject = f"Review started: {submission_title}"
    text_body = (
        f"Hello {student_name},\n\n"
        f"Professor {professor_name} has started reviewing your analysis "
        f"\"{submission_title}\"."
    )
    html_body = _wrap_html(
        "Professor started review",
        f"<p>Hello {student_name},</p>"
        f"<p>Professor <strong>{professor_name}</strong> has started reviewing "
        f"<strong>{submission_title}</strong>.</p>",
    )
    return subject, html_body, text_body


def review_completed_email(
    student_name: str,
    professor_name: str,
    submission_title: str,
    decision: str,
) -> tuple[str, str, str]:
    decision_label = decision.replace("_", " ").title()
    subject = f"Review completed: {submission_title}"
    text_body = (
        f"Hello {student_name},\n\n"
        f"Professor {professor_name} has completed the review of "
        f"\"{submission_title}\".\n\nDecision: {decision_label}"
    )
    html_body = _wrap_html(
        "Professor review completed",
        f"<p>Hello {student_name},</p>"
        f"<p>Professor <strong>{professor_name}</strong> has completed the review of "
        f"<strong>{submission_title}</strong>.</p>"
        f"<p><strong>Decision:</strong> {decision_label}</p>",
    )
    return subject, html_body, text_body


def submission_approved_email(
    student_name: str,
    professor_name: str,
    submission_title: str,
) -> tuple[str, str, str]:
    subject = f"Analysis approved: {submission_title}"
    text_body = (
        f"Hello {student_name},\n\n"
        f"Professor {professor_name} has approved your analysis "
        f"\"{submission_title}\"."
    )
    html_body = _wrap_html(
        "Analysis approved",
        f"<p>Hello {student_name},</p>"
        f"<p>Professor <strong>{professor_name}</strong> has approved "
        f"<strong>{submission_title}</strong>.</p>",
    )
    return subject, html_body, text_body
