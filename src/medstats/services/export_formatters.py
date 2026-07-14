"""Shared helpers for DOCX, PDF, and HTML export formatting."""
import re
from typing import Dict, Optional


def clean_section_heading(section: Dict) -> str:
    """Return display heading without 'Table No.X:' prefix."""
    content = (section.get("content") or "").strip()
    variable_name = section.get("variable_name") or ""

    cleaned = re.sub(r"^Table No\.\d+:\s*", "", content, flags=re.IGNORECASE).strip()
    if variable_name:
        return variable_name.replace("_", " ").title()
    if cleaned:
        return cleaned.title()
    return content


def is_major_heading(content: str) -> bool:
    """Headings that should start on a new page."""
    upper = (content or "").upper()
    return upper in (
        "STATISTICAL ANALYSIS",
        "STATISTICAL METHODOLOGY",
        "OBSERVATIONS AND RESULTS",
        "EXECUTIVE SUMMARY",
        "SUMMARY",
    )


def should_page_break_before(section_type: str, content: str = "", is_first_heading: bool = False) -> bool:
    """Each variable block and major section starts on a new page."""
    if section_type == "table_heading":
        return True
    if section_type == "heading" and is_major_heading(content) and not is_first_heading:
        return True
    return False
