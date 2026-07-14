"""Shared validators for scenario-based pipeline tests."""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Set


REQUIRED_REPORT_HEADINGS = {"OBSERVATIONS AND RESULTS", "Statistical Analysis"}
REQUIRED_VARIABLE_SECTION_TYPES = {"table_heading", "summary_text", "table"}


def collect_test_types(test_results: List[Dict]) -> Set[str]:
    types: Set[str] = set()
    for row in test_results:
        inner = row.get("test_result")
        if isinstance(inner, dict) and inner.get("test_type"):
            types.add(inner["test_type"])
        elif row.get("test_type"):
            types.add(row["test_type"])
    return types


def collect_variables_with_tests(test_results: List[Dict]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for row in test_results:
        var = row.get("variable_name")
        inner = row.get("test_result")
        if var and isinstance(inner, dict) and inner.get("test_type"):
            mapping[var.upper()] = inner["test_type"]
    return mapping


def group_sections_by_variable(sections: List[Dict]) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {}
    current = ""
    for section in sections:
        if section.get("type") == "table_heading":
            content = section.get("content") or ""
            match = re.search(r":\s*(.+)$", content, re.IGNORECASE)
            current = (match.group(1) if match else content).strip().upper()
            groups.setdefault(current, [])
            groups[current].append("table_heading")
        elif current:
            groups[current].append(section.get("type", ""))
    return groups


def assert_report_structure(
    sections: List[Dict],
    *,
    min_variables: int = 1,
    require_charts: bool = True,
) -> None:
    headings = {s.get("content") for s in sections if s.get("type") == "heading"}
    assert REQUIRED_REPORT_HEADINGS.issubset(headings), (
        f"Missing headings. Found: {headings}"
    )

    table_headings = [s for s in sections if s.get("type") == "table_heading"]
    assert len(table_headings) >= min_variables, (
        f"Expected at least {min_variables} variable blocks, got {len(table_headings)}"
    )

    by_var = group_sections_by_variable(sections)
    for var_name, types in by_var.items():
        missing = REQUIRED_VARIABLE_SECTION_TYPES - set(types)
        assert not missing, f"Variable {var_name} missing sections: {missing}"

    if require_charts:
        charts = [s for s in sections if s.get("type") == "chart"]
        assert len(charts) >= min(1, len(table_headings)), "Expected at least one chart section"

    # Table headings must follow Table No.X format
    for th in table_headings:
        content = th.get("content", "")
        assert re.match(r"Table No\.\d+:", content, re.IGNORECASE), (
            f"Invalid table heading format: {content}"
        )


def assert_charts_exist(charts: List[Dict]) -> None:
    assert charts, "No charts generated"
    for chart in charts:
        path = chart.get("path", "")
        assert path and os.path.exists(path), f"Chart file missing: {path}"


def assert_chart_suffixes(chart_paths: List[str], expected_suffixes: List[str]) -> None:
    """At least one chart path should contain each expected suffix fragment."""
    joined = " ".join(chart_paths).lower()
    for suffix in expected_suffixes:
        assert suffix.lower() in joined, (
            f"Expected chart type '{suffix}' in paths: {chart_paths}"
        )


def assert_expected_tests(
    test_results: List[Dict],
    expected: Dict[str, str],
    *,
    allow_alternate: Optional[Dict[str, List[str]]] = None,
) -> None:
    """expected maps VARIABLE_NAME -> test_type (case-insensitive variable keys)."""
    found = collect_variables_with_tests(test_results)
    allow_alternate = allow_alternate or {
        "chi_square": ["fisher_exact"],
        "fisher_exact": ["chi_square"],
        "unpaired_t_test": ["mann_whitney_u"],
        "mann_whitney_u": ["unpaired_t_test"],
        "one_way_anova": ["kruskal_wallis"],
        "kruskal_wallis": ["one_way_anova"],
    }

    for var, expected_type in expected.items():
        var_key = var.upper()
        assert var_key in found, f"Variable {var} has no inferential test. Found: {found}"
        actual = found[var_key]
        if actual == expected_type:
            continue
        alternates = allow_alternate.get(expected_type, [])
        assert actual in alternates, (
            f"{var}: expected {expected_type}, got {actual}"
        )


def assert_summary_json(summary: str) -> None:
    data = json.loads(summary)
    assert isinstance(data, dict)
    assert data.get("study_description"), "Summary missing study_description"
    assert data.get("key_findings"), "Summary missing key_findings"
    assert isinstance(data["key_findings"], list) and len(data["key_findings"]) >= 1
