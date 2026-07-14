"""
Review Agent — Tier 1: Mechanical/Clerical Checks.

Deterministic validations that require no LLM reasoning.
Covers variable coverage, percentage sums, table numbering,
narrative consistency, sample size, and chart coverage.
"""
import re
from typing import List, Optional

import pandas as pd

from src.medstats.agents.review_agent.schemas import ReviewItem, Severity


def _get_inner(tr: dict) -> dict:
    """Extract the inner test result, handling both nested and flat formats."""
    inner = tr.get("test_result")
    if inner is None or not isinstance(inner, dict):
        return tr
    return inner


def run_tier1_checks(data: dict, df: Optional[pd.DataFrame]) -> List[ReviewItem]:
    """Orchestrate all Tier 1 mechanical/clerical checks."""
    items = []

    items.extend(check_variable_coverage(data))
    items.extend(check_percentage_sums(data))
    items.extend(check_table_numbering(data))
    items.extend(check_narrative_consistency(data))
    items.extend(check_sample_size_consistency(data, df))
    items.extend(check_chart_coverage(data))

    return items


def check_variable_coverage(data: dict) -> List[ReviewItem]:
    """Ensure all input variables have corresponding report sections."""
    items = []
    variables = data.get("variables", [])
    sections = data.get("report_sections", [])
    test_results = data.get("test_results", [])

    reported_vars = set()
    for s in sections:
        if s.get("variable_name"):
            reported_vars.add(s["variable_name"])
    for tr in test_results:
        if tr.get("variable_name"):
            reported_vars.add(tr["variable_name"])

    skip_patterns = {"NO", "IPD", "NAME", "DATE", "DIAGNOSIS", "S.NO", "SR.NO"}

    for var in variables:
        name = var["name"]
        if name.upper() in skip_patterns:
            continue
        if name in reported_vars:
            items.append(ReviewItem(
                check_id="var_coverage",
                tier=1,
                severity=Severity.OK,
                category="Variable Coverage",
                message=f"{name} is included in report",
                variable=name,
            ))
        else:
            items.append(ReviewItem(
                check_id="var_missing",
                tier=1,
                severity=Severity.ERROR,
                category="Variable Coverage",
                message=f"Variable '{name}' from input data is NOT in the report",
                variable=name,
                suggestion="Add analysis for this variable or document why it was excluded",
                auto_fixable=True,
            ))
    return items


def check_percentage_sums(data: dict) -> List[ReviewItem]:
    """Verify frequency table percentages sum to ~100%."""
    items = []
    freq_tables = data.get("frequency_tables", [])

    for ft in freq_tables:
        var_name = ft.get("variable_name", "unknown")
        rows = ft.get("rows", [])
        data_rows = [r for r in rows if r.get("category") != "TOTAL"]

        if not data_rows:
            continue

        total_pct = sum(r.get("percentage", 0) for r in data_rows)
        if abs(total_pct - 100.0) < 1.0:
            items.append(ReviewItem(
                check_id="pct_sum",
                tier=1,
                severity=Severity.OK,
                category="Percentage Validation",
                message=f"{var_name}: percentages sum to {total_pct:.1f}%",
                variable=var_name,
            ))
        else:
            items.append(ReviewItem(
                check_id="pct_sum_error",
                tier=1,
                severity=Severity.WARNING,
                category="Percentage Validation",
                message=f"{var_name}: percentages sum to {total_pct:.1f}% (expected 100%)",
                variable=var_name,
                suggestion="Check for rounding errors or missing categories",
                auto_fixable=True,
            ))
    return items


def check_table_numbering(data: dict) -> List[ReviewItem]:
    """Verify table numbers are sequential."""
    sections = data.get("report_sections", [])
    table_numbers = []

    for s in sections:
        if s.get("type") == "table_heading":
            content = s.get("content", "")
            match = re.search(r"Table\s*No\.?\s*(\d+)", content, re.IGNORECASE)
            if match:
                table_numbers.append(int(match.group(1)))

    if not table_numbers:
        return []

    items = []
    expected = list(range(1, max(table_numbers) + 1))
    if table_numbers == expected:
        items.append(ReviewItem(
            check_id="table_seq",
            tier=1,
            severity=Severity.OK,
            category="Table Numbering",
            message=f"Tables numbered sequentially 1-{len(table_numbers)}",
        ))
    else:
        missing = set(expected) - set(table_numbers)
        items.append(ReviewItem(
            check_id="table_seq_error",
            tier=1,
            severity=Severity.WARNING,
            category="Table Numbering",
            message=f"Table numbering gaps: missing {missing}",
            suggestion="Renumber tables sequentially",
            auto_fixable=True,
        ))
    return items


def check_narrative_consistency(data: dict) -> List[ReviewItem]:
    """Check that narrative paragraphs reference correct numbers."""
    items = []
    sections = data.get("report_sections", [])
    sample_size = data.get("sample_size", 0)

    for s in sections:
        if s.get("type") == "paragraph":
            content = s.get("content", "")
            size_mentions = re.findall(r"(\d+)\s*subjects?", content)
            for mentioned in size_mentions:
                if int(mentioned) != sample_size and int(mentioned) != sample_size - 1:
                    items.append(ReviewItem(
                        check_id="sample_mismatch",
                        tier=1,
                        severity=Severity.WARNING,
                        category="Data Consistency",
                        message=f"Narrative mentions {mentioned} subjects but sample size is {sample_size}",
                        suggestion=f"Update to {sample_size} subjects",
                        auto_fixable=True,
                    ))

    if not any(i.check_id == "sample_mismatch" for i in items):
        items.append(ReviewItem(
            check_id="narrative_ok",
            tier=1,
            severity=Severity.OK,
            category="Data Consistency",
            message="Narrative numbers consistent with data",
        ))
    return items


def check_sample_size_consistency(data: dict, df: Optional[pd.DataFrame]) -> List[ReviewItem]:
    """Check sample size is consistent across sections."""
    items = []
    reported_size = data.get("sample_size", 0)

    if df is not None:
        actual_size = len(df.dropna(how="all"))
        if abs(actual_size - reported_size) > 2:
            items.append(ReviewItem(
                check_id="sample_size_drift",
                tier=1,
                severity=Severity.WARNING,
                category="Sample Size",
                message=f"Reported N={reported_size} but data has {actual_size} non-null rows",
                suggestion="Clarify inclusion/exclusion criteria",
            ))
        else:
            items.append(ReviewItem(
                check_id="sample_size_ok",
                tier=1,
                severity=Severity.OK,
                category="Sample Size",
                message=f"Sample size N={reported_size} confirmed",
            ))
    return items


def check_chart_coverage(data: dict) -> List[ReviewItem]:
    """Verify charts exist for reported variables."""
    items = []
    charts = data.get("charts", [])
    test_results = data.get("test_results", [])

    charted_vars = {c.get("variable", "") for c in charts}
    analyzed_vars = {tr.get("variable_name", "") for tr in test_results if tr.get("variable_name")}

    missing_charts = analyzed_vars - charted_vars - {""}
    if missing_charts:
        items.append(ReviewItem(
            check_id="missing_charts",
            tier=1,
            severity=Severity.INFO,
            category="Visualisation",
            message=f"{len(missing_charts)} variable(s) analyzed but no chart: {', '.join(list(missing_charts)[:3])}",
            suggestion="Consider adding charts for better visual representation",
        ))
    else:
        items.append(ReviewItem(
            check_id="charts_ok",
            tier=1,
            severity=Severity.OK,
            category="Visualisation",
            message=f"All {len(charts)} charts present for analyzed variables",
        ))
    return items
