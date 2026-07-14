"""
Review Agent — Tier 2: Statistical Methodology Checks.

Rule-based validation of statistical test selection, assumptions,
sample adequacy, normality, multiple testing, and chi-square assumptions.
"""
from typing import Dict, List, Optional

import pandas as pd

from src.medstats.agents.review_agent.schemas import ReviewItem, Severity


def _get_inner(tr: dict) -> dict:
    """Extract the inner test result, handling both nested and flat formats."""
    inner = tr.get("test_result")
    if inner is None or not isinstance(inner, dict):
        return tr
    return inner


def run_tier2_checks(data: dict, df: Optional[pd.DataFrame]) -> List[ReviewItem]:
    """Orchestrate all Tier 2 statistical methodology checks."""
    items = []

    items.extend(check_data_sufficiency(data, df))
    items.extend(check_test_appropriateness(data))
    items.extend(check_test_selection_logic(data, df))
    items.extend(check_sample_adequacy(data, df))
    items.extend(check_normality_assumptions(data, df))
    items.extend(check_multiple_testing(data))
    items.extend(check_chi_square_assumptions(data, df))

    return items


def check_data_sufficiency(data: dict, df: Optional[pd.DataFrame]) -> List[ReviewItem]:
    """Check if the masterchart data is sufficient for reliable statistical analysis."""
    items = []
    sample_size = data.get("sample_size", 0)
    variables = data.get("variables", [])

    if sample_size < 10:
        items.append(ReviewItem(
            check_id="sample_critically_small",
            tier=2,
            severity=Severity.ERROR,
            category="Data Sufficiency",
            message=f"Sample size N={sample_size} is critically small for any statistical inference",
            suggestion="Minimum recommended sample size is 30 for parametric tests, 20 for non-parametric. Collect more data.",
        ))
    elif sample_size < 20:
        items.append(ReviewItem(
            check_id="sample_very_small",
            tier=2,
            severity=Severity.WARNING,
            category="Data Sufficiency",
            message=f"Sample size N={sample_size} limits statistical power significantly",
            suggestion="Consider collecting more data (N\u226530 recommended). Use non-parametric tests and note limited power as a limitation.",
        ))

    n_vars = len([v for v in variables if v.get("dtype") == "numerical"])
    if n_vars > 0 and sample_size > 0:
        ratio = sample_size / n_vars
        if ratio < 5:
            items.append(ReviewItem(
                check_id="sample_var_ratio_critical",
                tier=2,
                severity=Severity.ERROR,
                category="Data Sufficiency",
                message=f"Only {ratio:.1f} observations per variable (N={sample_size}, {n_vars} numerical vars). Risk of spurious results.",
                suggestion=f"Need at least {n_vars * 10} observations for {n_vars} variables (rule: 10 per variable minimum)",
            ))
        elif ratio < 10:
            items.append(ReviewItem(
                check_id="sample_var_ratio_low",
                tier=2,
                severity=Severity.WARNING,
                category="Data Sufficiency",
                message=f"{ratio:.1f} observations per variable (recommended \u226510). Statistical power may be insufficient.",
                suggestion="Consider reducing variables or collecting more data",
            ))

    for var_info in variables:
        missing = var_info.get("missing_count", 0)
        if sample_size > 0 and missing > 0:
            missing_pct = (missing / sample_size) * 100
            var_name = var_info["name"]
            if missing_pct > 50:
                items.append(ReviewItem(
                    check_id="sample_high_missing",
                    tier=2,
                    severity=Severity.ERROR,
                    category="Data Sufficiency",
                    message=f"'{var_name}': {missing_pct:.0f}% missing data ({missing}/{sample_size}). Results unreliable.",
                    variable=var_name,
                    suggestion="Exclude this variable from analysis or collect complete data. Any test on this variable is underpowered.",
                ))
            elif missing_pct > 20:
                items.append(ReviewItem(
                    check_id="sample_moderate_missing",
                    tier=2,
                    severity=Severity.WARNING,
                    category="Data Sufficiency",
                    message=f"'{var_name}': {missing_pct:.0f}% missing data ({missing}/{sample_size}). May bias results.",
                    variable=var_name,
                    suggestion="Document missingness pattern. Consider if missing-at-random assumption holds.",
                ))

    if df is not None:
        actual_complete = len(df.dropna(how="all"))
        if actual_complete < sample_size * 0.7:
            items.append(ReviewItem(
                check_id="sample_effective_low",
                tier=2,
                severity=Severity.WARNING,
                category="Data Sufficiency",
                message=f"Effective sample (complete rows) is {actual_complete} vs reported N={sample_size}. {sample_size - actual_complete} rows have substantial missing data.",
                suggestion="Report effective sample size for each analysis. Consider listwise vs pairwise deletion impact.",
            ))

    return items


def check_test_appropriateness(data: dict) -> List[ReviewItem]:
    """Verify the right test was used for the data type."""
    items = []
    variables = {v["name"]: v for v in data.get("variables", [])}
    test_results = data.get("test_results", [])

    categorical_tests = {"z_test_proportion", "chi_square", "fisher_exact"}
    numerical_tests = {"unpaired_t_test", "paired_t_test", "mann_whitney_u", "one_way_anova", "kruskal_wallis", "shapiro_wilk"}

    for tr in test_results:
        var_name = tr.get("variable_name", "")
        inner = _get_inner(tr)
        test_type = inner.get("test_type", "") or tr.get("test_type", "")
        if not test_type or not var_name:
            continue

        var_info = variables.get(var_name)
        if not var_info:
            continue

        dtype = var_info.get("dtype", "")
        mismatch = False

        if dtype == "categorical" and test_type in numerical_tests:
            mismatch = True
        elif dtype == "numerical" and test_type in categorical_tests:
            mismatch = True

        if mismatch:
            items.append(ReviewItem(
                check_id="test_type_mismatch",
                tier=2,
                severity=Severity.ERROR,
                category="Test Appropriateness",
                message=f"{var_name}: {test_type} used for {dtype} data",
                variable=var_name,
                suggestion=f"Use {'chi-square/Fisher' if dtype == 'categorical' else 't-test/Mann-Whitney'} instead",
                auto_fixable=True,
            ))
        else:
            items.append(ReviewItem(
                check_id="test_type_ok",
                tier=2,
                severity=Severity.OK,
                category="Test Appropriateness",
                message=f"{var_name}: {test_type} appropriate for {dtype} data",
                variable=var_name,
            ))
    return items


def check_test_selection_logic(data: dict, df: Optional[pd.DataFrame]) -> List[ReviewItem]:
    """
    Deep validation of statistical test selection against actual data characteristics.
    Checks normality-based decisions, group counts, study design consistency, and
    whether parametric vs non-parametric was correctly chosen.
    """
    items = []
    variables = {v["name"]: v for v in data.get("variables", [])}
    test_results = data.get("test_results", [])
    study_design = data.get("study_design", "")
    sample_size = data.get("sample_size", 0)

    normality_results: Dict[str, bool] = {}
    for tr in test_results:
        inner = _get_inner(tr)
        if inner.get("test_type") == "shapiro_wilk":
            var_name = tr.get("variable_name", "") or inner.get("variable_name", "")
            p_val = inner.get("p_value")
            if var_name and p_val is not None and isinstance(p_val, (int, float)):
                normality_results[var_name] = p_val > 0.05

    var_categories: Dict[str, int] = {}
    if df is not None:
        for var_name, var_info in variables.items():
            if var_info.get("dtype") == "categorical" and var_name in df.columns:
                var_categories[var_name] = df[var_name].dropna().nunique()

    for tr in test_results:
        inner = _get_inner(tr)
        test_type = inner.get("test_type", "") or tr.get("test_type", "")
        var_name = tr.get("variable_name", "") or inner.get("variable_name", "")
        if not test_type or not var_name:
            continue

        var_info = variables.get(var_name)
        if not var_info:
            continue

        dtype = var_info.get("dtype", "")

        # Check 1: Parametric test on non-normal data
        if test_type in ("unpaired_t_test", "paired_t_test", "one_way_anova"):
            if var_name in normality_results and not normality_results[var_name]:
                non_param_alt = {
                    "unpaired_t_test": "Mann-Whitney U test",
                    "paired_t_test": "Wilcoxon signed-rank test",
                    "one_way_anova": "Kruskal-Wallis test",
                }
                items.append(ReviewItem(
                    check_id="parametric_on_nonnormal",
                    tier=2,
                    severity=Severity.ERROR,
                    category="Test Selection",
                    message=(
                        f"{var_name}: parametric test '{test_type}' used but "
                        f"Shapiro-Wilk indicates non-normal distribution"
                    ),
                    variable=var_name,
                    suggestion=f"Use {non_param_alt.get(test_type, 'non-parametric alternative')} instead",
                    auto_fixable=True,
                ))
            elif var_name in normality_results and normality_results[var_name]:
                items.append(ReviewItem(
                    check_id="parametric_normal_ok",
                    tier=2,
                    severity=Severity.OK,
                    category="Test Selection",
                    message=f"{var_name}: parametric '{test_type}' justified by normal distribution",
                    variable=var_name,
                ))

        # Check 2: Non-parametric test on normal data (missed power)
        if test_type in ("mann_whitney_u", "kruskal_wallis"):
            if var_name in normality_results and normality_results[var_name] and sample_size >= 30:
                param_alt = {
                    "mann_whitney_u": "unpaired t-test",
                    "kruskal_wallis": "one-way ANOVA",
                }
                items.append(ReviewItem(
                    check_id="nonparam_on_normal",
                    tier=2,
                    severity=Severity.INFO,
                    category="Test Selection",
                    message=(
                        f"{var_name}: non-parametric '{test_type}' used but data appears normal "
                        f"(N={sample_size}). Parametric test would have more statistical power."
                    ),
                    variable=var_name,
                    suggestion=f"Consider using {param_alt.get(test_type)} for higher power",
                ))

        # Check 3: t-test used when >2 groups exist
        if test_type in ("unpaired_t_test", "paired_t_test") and df is not None:
            cat_vars = [
                v for v in variables.values()
                if v.get("dtype") == "categorical" and v["name"] != var_name
            ]
            for cv in cat_vars:
                n_groups = var_categories.get(cv["name"], 0)
                if n_groups > 2:
                    items.append(ReviewItem(
                        check_id="ttest_multigroup",
                        tier=2,
                        severity=Severity.WARNING,
                        category="Test Selection",
                        message=(
                            f"{var_name}: t-test used but grouping variable "
                            f"'{cv['name']}' has {n_groups} categories (>2). "
                            f"T-test only valid for exactly 2-group comparison."
                        ),
                        variable=var_name,
                        suggestion="Use one-way ANOVA (parametric) or Kruskal-Wallis (non-parametric) for \u22653 groups",
                        auto_fixable=True,
                    ))
                    break

        # Check 4: ANOVA used with only 2 groups
        if test_type == "one_way_anova" and df is not None:
            cat_vars = [
                v for v in variables.values()
                if v.get("dtype") == "categorical" and v["name"] != var_name
            ]
            for cv in cat_vars:
                n_groups = var_categories.get(cv["name"], 0)
                if n_groups == 2:
                    items.append(ReviewItem(
                        check_id="anova_two_groups",
                        tier=2,
                        severity=Severity.INFO,
                        category="Test Selection",
                        message=(
                            f"{var_name}: ANOVA used but '{cv['name']}' has only 2 groups. "
                            f"While ANOVA works, t-test is more standard for 2-group comparison."
                        ),
                        variable=var_name,
                        suggestion="Consider using unpaired t-test for 2-group comparison (equivalent result, conventional choice)",
                    ))
                    break

        # Check 5: Chi-square on very small expected counts -> Fisher's exact
        if test_type == "chi_square" and df is not None and dtype == "categorical":
            if var_name in df.columns:
                counts = df[var_name].dropna().value_counts()
                n_categories = len(counts)
                min_expected = sample_size / n_categories if n_categories > 0 else 0
                if min_expected < 5 or (counts < 5).any():
                    items.append(ReviewItem(
                        check_id="chi_sq_should_fisher",
                        tier=2,
                        severity=Severity.WARNING,
                        category="Test Selection",
                        message=(
                            f"{var_name}: chi-square test used but expected cell frequencies <5. "
                            f"Chi-square approximation unreliable with sparse cells."
                        ),
                        variable=var_name,
                        suggestion="Use Fisher's exact test when any expected cell count < 5",
                        auto_fixable=True,
                    ))

        # Check 6: Z-test proportion appropriateness
        if test_type == "z_test_proportion":
            if dtype != "categorical":
                items.append(ReviewItem(
                    check_id="ztest_not_categorical",
                    tier=2,
                    severity=Severity.ERROR,
                    category="Test Selection",
                    message=f"{var_name}: Z-test for proportion used on non-categorical data",
                    variable=var_name,
                    suggestion="Z-test for proportions only applies to categorical/binary variables",
                ))
            elif dtype == "categorical" and var_name in var_categories:
                n_cats = var_categories[var_name]
                if n_cats > 5:
                    items.append(ReviewItem(
                        check_id="ztest_too_many_cats",
                        tier=2,
                        severity=Severity.INFO,
                        category="Test Selection",
                        message=(
                            f"{var_name}: Z-test proportion applied to variable with {n_cats} "
                            f"categories. Z-test is most meaningful for binary/few-category variables."
                        ),
                        variable=var_name,
                        suggestion="Consider chi-square goodness-of-fit test for multi-category variables",
                    ))

        # Check 7: Study design consistency
        if study_design == "single_cohort":
            two_group_tests = {"unpaired_t_test", "mann_whitney_u", "chi_square"}
            if test_type in two_group_tests:
                items.append(ReviewItem(
                    check_id="study_design_mismatch",
                    tier=2,
                    severity=Severity.WARNING,
                    category="Test Selection",
                    message=(
                        f"{var_name}: comparison test '{test_type}' used in single-cohort study design. "
                        f"This requires defining comparison groups."
                    ),
                    variable=var_name,
                    suggestion="Verify that the study design should be 'two-group' or 'multi-group' instead of single cohort, or clarify which groups are being compared",
                ))

        # Check 8: Small sample (N<30) with parametric test and no normality check
        if test_type in ("unpaired_t_test", "paired_t_test") and sample_size < 30:
            if var_name not in normality_results:
                items.append(ReviewItem(
                    check_id="small_n_no_normality",
                    tier=2,
                    severity=Severity.WARNING,
                    category="Test Selection",
                    message=(
                        f"{var_name}: parametric '{test_type}' with small sample (N={sample_size}) "
                        f"but no normality test performed. Central Limit Theorem unreliable for N<30."
                    ),
                    variable=var_name,
                    suggestion="Run Shapiro-Wilk test first; if non-normal, switch to Mann-Whitney U / Wilcoxon",
                    auto_fixable=True,
                ))

    return items


def check_sample_adequacy(data: dict, df: Optional[pd.DataFrame]) -> List[ReviewItem]:
    """Check if sample size is adequate for chosen tests."""
    items = []
    sample_size = data.get("sample_size", 0)
    test_results = data.get("test_results", [])

    min_sizes = {
        "unpaired_t_test": 30,
        "paired_t_test": 15,
        "one_way_anova": 20,
        "chi_square": 20,
    }

    for tr in test_results:
        inner = _get_inner(tr)
        test_type = inner.get("test_type", "")
        min_n = min_sizes.get(test_type)

        if min_n and sample_size < min_n:
            items.append(ReviewItem(
                check_id="sample_too_small",
                tier=2,
                severity=Severity.WARNING,
                category="Sample Adequacy",
                message=f"N={sample_size} may be insufficient for {test_type} (recommended \u2265{min_n})",
                suggestion="Consider non-parametric alternative or note as limitation",
            ))

    if not any(i.check_id == "sample_too_small" for i in items):
        items.append(ReviewItem(
            check_id="sample_adequate",
            tier=2,
            severity=Severity.OK,
            category="Sample Adequacy",
            message=f"Sample size N={sample_size} adequate for all tests used",
        ))
    return items


def check_normality_assumptions(data: dict, df: Optional[pd.DataFrame]) -> List[ReviewItem]:
    """Verify normality was tested before parametric tests."""
    items = []
    test_results = data.get("test_results", [])

    parametric_tests = {"unpaired_t_test", "paired_t_test", "one_way_anova"}
    parametric_vars = set()
    normality_tested = set()

    for tr in test_results:
        var_name = tr.get("variable_name", "")
        inner = _get_inner(tr)
        test_type = inner.get("test_type", "")

        if test_type in parametric_tests:
            parametric_vars.add(var_name)
        if test_type == "shapiro_wilk":
            normality_tested.add(var_name)

    untested = parametric_vars - normality_tested
    if untested:
        for var in untested:
            items.append(ReviewItem(
                check_id="normality_not_tested",
                tier=2,
                severity=Severity.WARNING,
                category="Assumptions",
                message=f"Parametric test used for '{var}' without documented normality check",
                variable=var,
                suggestion="Add Shapiro-Wilk test or state assumption justification",
            ))
    else:
        items.append(ReviewItem(
            check_id="normality_ok",
            tier=2,
            severity=Severity.OK,
            category="Assumptions",
            message="Normality assumptions verified for all parametric tests",
        ))
    return items


def check_multiple_testing(data: dict) -> List[ReviewItem]:
    """Flag if many comparisons without correction."""
    items = []
    test_results = data.get("test_results", [])

    significance_tests = [
        tr for tr in test_results
        if _get_inner(tr).get("p_value") is not None
    ]

    if len(significance_tests) > 5:
        significant_count = sum(
            1 for tr in significance_tests
            if _get_inner(tr).get("is_significant_05")
        )
        items.append(ReviewItem(
            check_id="multiple_testing",
            tier=2,
            severity=Severity.INFO,
            category="Multiple Testing",
            message=f"{len(significance_tests)} tests performed, {significant_count} significant. Consider Bonferroni correction.",
            suggestion=f"Adjusted \u03b1 = {0.05 / len(significance_tests):.4f} (Bonferroni)",
        ))
    return items


def check_chi_square_assumptions(data: dict, df: Optional[pd.DataFrame]) -> List[ReviewItem]:
    """Check expected cell counts for chi-square tests."""
    items = []
    if df is None:
        return items

    variables = {v["name"]: v for v in data.get("variables", [])}
    test_results = data.get("test_results", [])

    for tr in test_results:
        inner = _get_inner(tr)
        test_type = inner.get("test_type", "")
        var_name = tr.get("variable_name", "")

        if test_type != "chi_square" or not var_name:
            continue

        var_info = variables.get(var_name)
        if not var_info or var_info.get("dtype") != "categorical":
            continue

        if var_name in df.columns:
            value_counts = df[var_name].dropna().value_counts()
            min_count = value_counts.min() if len(value_counts) > 0 else 0

            if min_count < 5:
                items.append(ReviewItem(
                    check_id="chi_sq_small_cell",
                    tier=2,
                    severity=Severity.WARNING,
                    category="Chi-square Assumptions",
                    message=f"{var_name}: smallest category has only {min_count} cases (expected \u22655 for chi-square)",
                    variable=var_name,
                    suggestion="Consider Fisher's exact test or merge sparse categories",
                    auto_fixable=True,
                ))
    return items
