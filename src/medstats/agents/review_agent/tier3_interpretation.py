"""
Review Agent — Tier 3: Clinical Interpretation checks.

LLM-powered interpretation validation with heuristic fallbacks.
"""
from typing import List, Dict, Optional

import asyncio
import pandas as pd

from src.medstats.agents.review_agent.schemas import ReviewItem, Severity
from src.medstats.agents.review_agent.prompts import REVIEW_SYSTEM_PROMPT, INTERPRETATION_CHECK_TEMPLATE
from src.shared.logging.logger import get_logger

logger = get_logger(__name__)


def _get_inner(tr: dict) -> dict:
    """Extract the inner test result, handling both nested and flat formats."""
    inner = tr.get("test_result")
    if inner is None or not isinstance(inner, dict):
        return tr
    return inner


async def run_tier3_checks(data: dict, df: Optional[pd.DataFrame], services=None) -> List[ReviewItem]:
    """Run all Tier 3 interpretation checks."""
    items = []

    items.extend(check_clinical_significance(data))
    items.extend(check_negative_results(data))

    if services:
        tier3_llm = await llm_interpretation_check(data, services)
        items.extend(tier3_llm)

    return items


def check_clinical_significance(data: dict) -> List[ReviewItem]:
    """Flag cases where statistical significance may not equal clinical significance."""
    items = []
    test_results = data.get("test_results", [])
    sample_size = data.get("sample_size", 0)

    for tr in test_results:
        inner = _get_inner(tr)
        p_value = inner.get("p_value")
        is_sig = inner.get("is_significant_05", False)
        var_name = tr.get("variable_name", "")

        if not p_value or not is_sig:
            continue
        if not isinstance(p_value, (int, float)):
            continue

        if sample_size > 200 and p_value > 0.01:
            items.append(ReviewItem(
                check_id="clinical_vs_statistical",
                tier=3,
                severity=Severity.INFO,
                category="Clinical Significance",
                message=f"{var_name}: statistically significant (p={p_value:.4f}) with large N={sample_size}. Verify clinical relevance.",
                variable=var_name,
                suggestion="Report effect size and comment on clinical meaningfulness",
            ))
    return items


def check_negative_results(data: dict) -> List[ReviewItem]:
    """Check proper interpretation of non-significant results."""
    items = []
    sections = data.get("report_sections", [])

    problematic_phrases = [
        "no difference exists",
        "proved that there is no",
        "confirmed no association",
        "ruled out",
    ]

    for s in sections:
        content = s.get("content", "").lower()
        for phrase in problematic_phrases:
            if phrase in content:
                items.append(ReviewItem(
                    check_id="negative_overstatement",
                    tier=3,
                    severity=Severity.WARNING,
                    category="Interpretation",
                    message=f"Potential overstatement of negative result: '{phrase}' found",
                    suggestion="Non-significant p-value means 'insufficient evidence to reject H0', not 'no difference'",
                ))
                break

    test_results = data.get("test_results", [])
    for tr in test_results:
        inner = _get_inner(tr)
        p_value = inner.get("p_value")
        var_name = tr.get("variable_name", "")
        if p_value and isinstance(p_value, (int, float)) and 0.05 < p_value < 0.10:
            items.append(ReviewItem(
                check_id="borderline_p",
                tier=3,
                severity=Severity.INFO,
                category="Interpretation",
                message=f"{var_name}: borderline p-value ({p_value:.4f}). Consider noting as 'trend'.",
                variable=var_name,
                suggestion="Mention as 'approaching significance' or 'trend towards significance (p=X)'",
            ))
    return items


async def llm_interpretation_check(data: dict, services) -> List[ReviewItem]:
    """Use LLM to check if conclusions are supported by evidence."""
    items = []
    sections = data.get("report_sections", [])
    title = data.get("title", "")

    narratives = [s["content"] for s in sections if s.get("type") == "paragraph"]
    test_summaries = []
    for tr in data.get("test_results", []):
        inner = _get_inner(tr)
        if inner.get("significance_statement"):
            test_summaries.append(f"- {tr.get('variable_name')}: {inner['significance_statement']}")

    if not narratives or not test_summaries:
        return items

    prompt = INTERPRETATION_CHECK_TEMPLATE.format(
        title=title,
        test_summaries="\n".join(test_summaries[:10]),
        narratives="\n".join(narratives[:5]),
    )

    try:
        response = await asyncio.wait_for(
            services.generate_json(
                prompt=prompt,
                system_prompt=REVIEW_SYSTEM_PROMPT,
                temperature=0.2,
            ),
            timeout=45.0,
        )
        if response and response.get("issues"):
            for issue in response["issues"][:5]:
                items.append(ReviewItem(
                    check_id="llm_interpretation",
                    tier=3,
                    severity=Severity.WARNING,
                    category="Interpretation (AI)",
                    message=issue.get("issue", ""),
                    variable=issue.get("variable", ""),
                    suggestion=issue.get("suggestion", ""),
                ))
    except asyncio.TimeoutError:
        logger.warning("LLM interpretation check timed out after 45s")
    except Exception as e:
        logger.warning("LLM interpretation check failed", error=str(e))

    return items
