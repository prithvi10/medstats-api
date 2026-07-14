"""
Report Generator Agent — Formatting utilities.

Handles high-cardinality grouping, methodology note generation,
and template-based narrative fallbacks.
"""
from typing import List, Dict, Set

from src.medstats.agents.report_generator.schemas import TEST_NAME_MAP


def group_top_categories(rows: list, top_n: int = 7) -> list:
    """Group low-frequency categories into 'Others' for readable output."""
    sorted_rows = sorted(rows, key=lambda r: r["count"], reverse=True)
    top = sorted_rows[:top_n]
    others = sorted_rows[top_n:]

    if others:
        others_count = sum(r["count"] for r in others)
        total = sum(r["count"] for r in rows)
        others_pct = (others_count / total * 100) if total > 0 else 0
        top.append({
            "category": f"Others ({len(others)} categories)",
            "count": others_count,
            "percentage": round(others_pct, 2),
        })

    return top


def generate_methodology_note(tests_used: Set[str], sample_size: int) -> str:
    """Generate the statistical analysis methodology paragraph."""
    named_tests = [TEST_NAME_MAP.get(t, t) for t in tests_used]
    tests_str = ", ".join(named_tests) if named_tests else "descriptive statistics"

    return (
        f"Statistical analysis was done by descriptive statistics as mean, SD, "
        f"percentage/proportions. Comparison of variables was done by applying "
        f"{tests_str} at 5% (p<0.05) and 1% (p<0.01) level of significance. "
        f"Statistical analysis was performed using MedStats AI platform with "
        f"scipy and statsmodels libraries for Python."
    )


def generate_narrative_template(
    var_name: str, descriptive: dict, freq_table, sample_size: int
) -> str:
    """Fallback template-based narrative when LLM is unavailable."""
    if not descriptive:
        return f"The distribution of {var_name.lower()} is presented in the table below."
    if descriptive.get("mean") is not None:
        return (
            f"The {var_name.lower()} ranged from {descriptive.get('min_val')} to "
            f"{descriptive.get('max_val')} with a mean of {descriptive['mean']} "
            f"and standard deviation of {descriptive.get('std', 'N/A')}."
        )
    elif freq_table and isinstance(freq_table, dict) and freq_table.get("rows"):
        rows = [r for r in freq_table["rows"] if r.get("category") != "TOTAL"]
        if rows:
            top = rows[0]
            return (
                f"Among {sample_size} subjects, the majority ({top['percentage']:.1f}%) "
                f"had {top['category'].lower()} as the {var_name.lower()}."
            )
    return f"The distribution of {var_name.lower()} is presented in the table below."
