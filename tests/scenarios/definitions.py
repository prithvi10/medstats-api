"""Scenario catalog for student local tests."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


SCENARIOS_ROOT = Path(__file__).parent.parent / "fixtures" / "scenarios"


@dataclass
class StudentScenario:
    id: str
    title: str
    intent: str
    folder: str
    data_file: str = "data.xlsx"
    intent_file: str = "intent.docx"
    study_design: str = "single_cohort"
    expected_tests: Dict[str, str] = field(default_factory=dict)
    expected_chart_suffixes: List[str] = field(default_factory=list)
    min_variables: int = 1
    min_sample_size: int = 20
    use_masterchart: bool = False

    @property
    def path(self) -> Path:
        return SCENARIOS_ROOT / self.folder

    def data_path(self) -> Path:
        return self.path / self.data_file

    def intent_path(self) -> Path:
        return self.path / self.intent_file


STUDENT_SCENARIOS: List[StudentScenario] = [
    StudentScenario(
        id="01_single_cohort_binary",
        title="Sex Distribution in Paediatric Fracture Cohort",
        folder="01_single_cohort_binary",
        intent=(
            "A prospective observational cohort study evaluating demographic patterns "
            "in children with long bone fractures. Assess whether one sex predominates."
        ),
        study_design="single_cohort",
        expected_tests={"SEX": "z_test_proportion"},
        expected_chart_suffixes=["donut"],
        min_variables=2,
    ),
    StudentScenario(
        id="02_single_cohort_multicategory",
        title="Mechanism of Injury Distribution",
        folder="02_single_cohort_multicategory",
        intent=(
            "Prospective cohort describing injury mechanisms in orthopaedic trauma patients. "
            "Determine whether any mechanism is statistically predominant."
        ),
        study_design="single_cohort",
        expected_tests={"MECHANISM OF INJURY": "z_test_proportion"},
        expected_chart_suffixes=["pie"],
        min_variables=2,
    ),
    StudentScenario(
        id="03_single_cohort_numerical",
        title="Age Distribution in Fracture Patients",
        folder="03_single_cohort_numerical",
        intent=(
            "Observational cohort assessing age distribution and normality of age among "
            "paediatric fracture patients."
        ),
        study_design="single_cohort",
        expected_tests={},  # shapiro only — no var-level inferential block required
        expected_chart_suffixes=["hist"],
        min_variables=2,
    ),
    StudentScenario(
        id="04_single_cohort_ordered_ranges",
        title="Age Band Distribution",
        folder="04_single_cohort_ordered_ranges",
        intent=(
            "Prospective cohort reporting ordered age bands (6-7, 8-9, 10-11, etc.) "
            "for fracture patients."
        ),
        study_design="single_cohort",
        expected_tests={"AGE BAND": "z_test_proportion"},
        expected_chart_suffixes=["vbar"],
        min_variables=2,
    ),
    StudentScenario(
        id="05_two_group_ttest",
        title="Recovery Days: Control vs Intervention",
        folder="05_two_group_ttest",
        intent=(
            "Randomized two-group study comparing recovery days between control and "
            "intervention arms after fracture management."
        ),
        study_design="two_group",
        expected_tests={"RECOVERY DAYS": "unpaired_t_test"},
        expected_chart_suffixes=["donut", "hist"],
        min_variables=2,
    ),
    StudentScenario(
        id="06_two_group_mann_whitney",
        title="Hospital Stay: Skewed Comparison",
        folder="06_two_group_mann_whitney",
        intent=(
            "Randomized trial comparing length of hospital stay between two treatment groups. "
            "Stay data are expected to be non-normally distributed."
        ),
        study_design="two_group",
        expected_tests={"HOSPITAL STAY": "mann_whitney_u"},
        expected_chart_suffixes=["hist", "donut"],
        min_variables=2,
    ),
    StudentScenario(
        id="07_two_group_chi_square",
        title="Recovery Outcome by Group",
        folder="07_two_group_chi_square",
        intent=(
            "Two-group comparative study evaluating whether recovery outcome (Yes/No) "
            "differs between control and treatment."
        ),
        study_design="two_group",
        expected_tests={"RECOVERED": "chi_square"},
        expected_chart_suffixes=["donut"],
        min_variables=2,
    ),
    StudentScenario(
        id="08_two_group_fisher_exact",
        title="Small-Sample Complication Rate",
        folder="08_two_group_fisher_exact",
        intent=(
            "Small-sample randomized comparison of complication rates between two groups."
        ),
        study_design="two_group",
        expected_tests={"COMPLICATION": "fisher_exact"},
        expected_chart_suffixes=["donut"],
        min_variables=2,
        min_sample_size=16,
    ),
    StudentScenario(
        id="09_multi_group_anova",
        title="Pain Score Across Three Treatments",
        folder="09_multi_group_anova",
        intent=(
            "Multi-arm clinical study comparing pain scores across three treatment groups."
        ),
        study_design="multi_group",
        expected_tests={"PAIN SCORE": "one_way_anova"},
        expected_chart_suffixes=["pie", "hist"],
        min_variables=2,
    ),
    StudentScenario(
        id="10_multi_group_kruskal",
        title="Complication Score Across Three Arms",
        folder="10_multi_group_kruskal",
        intent=(
            "Three-arm study with skewed complication scores compared across treatment groups."
        ),
        study_design="multi_group",
        expected_tests={"COMPLICATION SCORE": "kruskal_wallis"},
        expected_chart_suffixes=["pie", "hist"],
        min_variables=2,
    ),
    StudentScenario(
        id="11_complex_charts",
        title="Chart Type Matrix Scenario",
        folder="11_complex_charts",
        intent=(
            "Prospective cohort with multiple categorical and continuous variables designed "
            "to exercise pie, donut, bar, lollipop, histogram, and vertical bar charts."
        ),
        study_design="single_cohort",
        expected_tests={"SEX": "z_test_proportion", "INJURY TYPE": "z_test_proportion"},
        expected_chart_suffixes=["donut", "pie", "hist", "vbar", "bar"],
        min_variables=5,
    ),
    StudentScenario(
        id="12_masterchart_docx",
        title="Master Chart DOCX Ingestion",
        folder="12_masterchart_docx",
        data_file="masterchart.docx",
        intent=(
            "Prospective observational study parsed from a Word master chart with embedded tables."
        ),
        study_design="single_cohort",
        expected_tests={"SEX": "z_test_proportion"},
        expected_chart_suffixes=["donut", "bar"],
        min_variables=3,
        use_masterchart=True,
    ),
    StudentScenario(
        id="13_medicine5_reference",
        title="Reference Thesis Dataset (Medicine 5)",
        folder="13_medicine5_reference",
        data_file="medicine5_data.xlsx",
        intent_file="discussion.docx",
        intent=(
            "Evaluate outcome of titanium elastic nail fixation for paediatric long bone fractures."
        ),
        study_design="single_cohort",
        expected_tests={"SEX": "z_test_proportion"},
        expected_chart_suffixes=["donut", "pie", "bar"],
        min_variables=5,
        min_sample_size=45,
    ),
    StudentScenario(
        id="14_complex_analysis",
        title="Complex Multi-Variable Clinical Analysis",
        folder="14_complex_analysis",
        intent=(
            "Randomized two-group study with demographics, outcomes, and complications. "
            "Comprehensive statistical analysis across numerical and categorical endpoints."
        ),
        study_design="two_group",
        expected_tests={
            "PAIN SCORE": "unpaired_t_test",
            "RECOVERED": "chi_square",
            "COMPLICATION": "chi_square",
        },
        expected_chart_suffixes=["donut", "hist"],
        min_variables=5,
    ),
]

SCENARIO_BY_ID = {s.id: s for s in STUDENT_SCENARIOS}
