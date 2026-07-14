"""Preset feedback codes for structured professor reviews."""

REVIEW_PRESETS = {
    "methodology": [
        {"code": "wrong_test", "label": "Wrong test selected", "hint": "Consider an alternative statistical test for this variable."},
        {"code": "missing_normality", "label": "Missing normality check", "hint": "Add or document a normality assessment before parametric tests."},
        {"code": "sample_size_low", "label": "Sample size concern", "hint": "Sample may be too small for the chosen test; note limitations."},
        {"code": "multiple_testing", "label": "Multiple testing issue", "hint": "Address multiple comparisons or Bonferroni adjustment."},
        {"code": "study_design_mismatch", "label": "Study design mismatch", "hint": "Tests do not match the stated study design."},
        {"code": "interpretation_overreach", "label": "Over-interpretation", "hint": "Tone down causal or overstated claims in interpretation."},
    ],
    "executive_summary": [
        {"code": "unclear_wording", "label": "Unclear wording", "hint": "Rewrite for clarity in academic English."},
        {"code": "add_limitations", "label": "Add limitations", "hint": "Include study limitations in the summary."},
        {"code": "strengthen_conclusion", "label": "Strengthen conclusion", "hint": "Make conclusions better aligned with findings."},
        {"code": "cite_assumptions", "label": "State assumptions", "hint": "Explicitly mention statistical assumptions used."},
    ],
    "variable": [
        {"code": "rerun_descriptive", "label": "Improve descriptive stats", "hint": "Expand or correct descriptive statistics for this variable."},
        {"code": "add_ci", "label": "Add confidence interval", "hint": "Report confidence intervals where applicable."},
        {"code": "clarify_groups", "label": "Clarify group labels", "hint": "Make group definitions and labels clearer."},
    ],
}

# Map preset codes to correction instructions for RevisionResolver
PRESET_INSTRUCTIONS = {
    "wrong_test": "Review and correct the statistical test selection.",
    "missing_normality": "Add normality testing documentation before parametric analysis.",
    "sample_size_low": "Acknowledge sample size limitations in the methodology and interpretation.",
    "multiple_testing": "Address multiple testing correction in the methodology section.",
    "study_design_mismatch": "Align statistical tests with the stated study design.",
    "interpretation_overreach": "Revise interpretation to avoid overstated conclusions.",
    "unclear_wording": "Rewrite unclear narrative sections in clearer academic English.",
    "add_limitations": "Add a limitations paragraph to the executive summary.",
    "strengthen_conclusion": "Strengthen the conclusion to match the statistical findings.",
    "cite_assumptions": "Explicitly state statistical assumptions in the methodology.",
    "rerun_descriptive": "Improve descriptive statistics presentation for this variable.",
    "add_ci": "Include confidence intervals in the results narrative.",
    "clarify_groups": "Clarify group definitions and comparison labels.",
}

VALID_TESTS_BY_DESIGN = {
    "single_cohort": [
        "z_test_proportion", "chi_square", "fisher_exact", "shapiro_wilk",
    ],
    "two_group": [
        "unpaired_t_test", "paired_t_test", "mann_whitney_u",
        "chi_square", "fisher_exact", "shapiro_wilk", "z_test_proportion",
    ],
    "multi_group": [
        "one_way_anova", "kruskal_wallis", "chi_square",
        "shapiro_wilk", "z_test_proportion",
    ],
}

ALL_TEST_OPTIONS = [
    {"value": "z_test_proportion", "label": "Z-test of proportions"},
    {"value": "chi_square", "label": "Chi-square test"},
    {"value": "fisher_exact", "label": "Fisher's exact test"},
    {"value": "unpaired_t_test", "label": "Student's unpaired t-test"},
    {"value": "paired_t_test", "label": "Student's paired t-test"},
    {"value": "mann_whitney_u", "label": "Mann-Whitney U test"},
    {"value": "one_way_anova", "label": "One-way ANOVA"},
    {"value": "kruskal_wallis", "label": "Kruskal-Wallis test"},
    {"value": "shapiro_wilk", "label": "Shapiro-Wilk test"},
]
