"""
Report Generator Agent — Schemas and data models.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ReportSection:
    """A single section in the generated report."""

    type: str  # heading, table_heading, paragraph, descriptive_note, test_result, table, chart
    content: str = ""
    variable_name: str = ""
    headers: List[str] = field(default_factory=list)
    rows: List[List[str]] = field(default_factory=list)
    chart_path: str = ""


@dataclass
class ChartSpec:
    """Specification for a generated chart."""

    variable: str
    path: str
    table_number: int = 0
    chart_type: str = ""  # pie, bar, grouped_bar, line


TEST_NAME_MAP = {
    "z_test_proportion": "Z-test of proportions",
    "chi_square": "Chi-square test",
    "fisher_exact": "Fisher's exact test",
    "unpaired_t_test": "Student's unpaired t-test",
    "paired_t_test": "Student's paired t-test",
    "mann_whitney_u": "Mann-Whitney U test",
    "one_way_anova": "One-way ANOVA",
    "kruskal_wallis": "Kruskal-Wallis test",
    "shapiro_wilk": "Shapiro-Wilk test",
}

TEST_RATIONALE_MAP = {
    "z_test_proportion": "to compare observed proportions against expected values for categorical variables",
    "chi_square": "to determine the association between categorical variables",
    "fisher_exact": "to test association when sample sizes are small or expected frequencies are below 5",
    "unpaired_t_test": "to compare means between two independent groups for normally distributed continuous data",
    "paired_t_test": "to compare paired observations or pre-post measurements",
    "mann_whitney_u": "to compare distributions between two groups when normality assumptions are not met",
    "one_way_anova": "to compare means across three or more independent groups",
    "kruskal_wallis": "to compare distributions across multiple groups for non-parametric data",
    "shapiro_wilk": "to assess whether the data follows a normal distribution",
}
