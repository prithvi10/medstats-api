"""
Hardcoded statistical knowledge graph bootstrap.

Covers all tests in TEST_NAME_MAP, common assumptions, and ICMR guideline pointers.
Loaded once via build_rag_corpus.py before PDF ingestion.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

NodeSpec = Dict[str, Any]
EdgeSpec = Dict[str, Any]


def get_seed_nodes() -> List[NodeSpec]:
    """Return ~50 knowledge graph nodes."""
    tests = [
        ("Z-test of proportions", "statistical_test", ["z-test", "z test proportion"], {
            "test_key": "z_test_proportion",
            "applicable_designs": ["single_cohort", "two_group"],
            "data_types": ["categorical"],
        }),
        ("Chi-square test", "statistical_test", ["chi-square", "chi square"], {
            "test_key": "chi_square",
            "applicable_designs": ["single_cohort", "two_group", "multi_group"],
            "data_types": ["categorical"],
        }),
        ("Fisher's exact test", "statistical_test", ["fisher exact", "fisher's exact"], {
            "test_key": "fisher_exact",
            "applicable_designs": ["single_cohort", "two_group"],
            "data_types": ["categorical"],
        }),
        ("Student's unpaired t-test", "statistical_test", ["unpaired t-test", "independent t-test"], {
            "test_key": "unpaired_t_test",
            "applicable_designs": ["two_group"],
            "data_types": ["numerical"],
            "requires_normality": True,
        }),
        ("Student's paired t-test", "statistical_test", ["paired t-test"], {
            "test_key": "paired_t_test",
            "applicable_designs": ["two_group"],
            "data_types": ["numerical"],
            "requires_normality": True,
        }),
        ("Mann-Whitney U test", "statistical_test", ["mann-whitney", "wilcoxon rank-sum"], {
            "test_key": "mann_whitney_u",
            "applicable_designs": ["two_group"],
            "data_types": ["numerical", "ordinal"],
            "requires_normality": False,
        }),
        ("One-way ANOVA", "statistical_test", ["anova", "one-way anova"], {
            "test_key": "one_way_anova",
            "applicable_designs": ["multi_group"],
            "data_types": ["numerical"],
            "requires_normality": True,
        }),
        ("Kruskal-Wallis test", "statistical_test", ["kruskal-wallis"], {
            "test_key": "kruskal_wallis",
            "applicable_designs": ["multi_group"],
            "data_types": ["numerical", "ordinal"],
            "requires_normality": False,
        }),
        ("Shapiro-Wilk test", "statistical_test", ["shapiro-wilk", "normality test"], {
            "test_key": "shapiro_wilk",
            "applicable_designs": ["single_cohort", "two_group", "multi_group"],
            "data_types": ["numerical"],
        }),
    ]

    assumptions = [
        ("Normality assumption", "assumption", ["normal distribution", "gaussian"], {}),
        ("Independence of observations", "assumption", ["independent samples"], {}),
        ("Homogeneity of variance", "assumption", ["equal variance", "homoscedasticity"], {}),
        ("Expected cell frequency >= 5", "assumption", ["expected frequency", "cell count"], {}),
    ]

    concepts = [
        ("p-value", "concept", ["p value", "significance level"], {"alpha": 0.05}),
        ("Type I error", "concept", ["alpha error", "false positive"], {}),
        ("Type II error", "concept", ["beta error", "false negative"], {}),
        ("Statistical power", "concept", ["power analysis"], {}),
        ("Bonferroni correction", "concept", ["multiple testing correction"], {}),
        ("Confidence interval", "concept", ["95% CI"], {}),
        ("Effect size", "concept", ["cohen d", "odds ratio"], {}),
    ]

    designs = [
        ("Single cohort study", "study_design", ["single group", "descriptive"], {"design_key": "single_cohort"}),
        ("Two-group comparison", "study_design", ["two group", "comparative"], {"design_key": "two_group"}),
        ("Multi-group comparison", "study_design", ["multi group", "three arms"], {"design_key": "multi_group"}),
    ]

    properties = [
        ("Small sample (N<30)", "dataset_property", ["small sample", "n less than 30"], {"threshold": 30}),
        ("Small sample (N<20)", "dataset_property", ["very small sample"], {"threshold": 20}),
        ("Categorical variable", "dataset_property", ["categorical data", "nominal"], {}),
        ("Numerical variable", "dataset_property", ["continuous data", "quantitative"], {}),
        ("Ordinal variable", "dataset_property", ["ordinal data", "ranked"], {}),
        ("Non-normal distribution", "dataset_property", ["skewed", "non-normal"], {}),
        ("Sparse contingency table", "dataset_property", ["expected cell < 5"], {}),
    ]

    guidelines = [
        ("ICMR Statistical Guidelines 2022", "guideline_source", ["ICMR", "Indian Council of Medical Research"], {}),
        ("CONSORT 2010", "guideline_source", ["CONSORT", "RCT reporting"], {}),
        ("STROBE", "guideline_source", ["STROBE", "observational studies"], {}),
    ]

    nodes: List[NodeSpec] = []
    for name, etype, aliases, props in tests + assumptions + concepts + designs + properties + guidelines:
        nodes.append({
            "name": name,
            "entity_type": etype,
            "aliases": aliases,
            "description": f"{etype.replace('_', ' ').title()}: {name}",
            "properties": props,
        })
    return nodes


def get_seed_edges() -> List[EdgeSpec]:
    """Return ~150 knowledge graph edges."""
    edges: List[Tuple[str, str, str, float]] = [
        # Parametric tests require normality
        ("Student's unpaired t-test", "Normality assumption", "requires", 1.0),
        ("Student's paired t-test", "Normality assumption", "requires", 1.0),
        ("One-way ANOVA", "Normality assumption", "requires", 1.0),
        ("One-way ANOVA", "Homogeneity of variance", "requires", 1.0),
        ("Student's unpaired t-test", "Homogeneity of variance", "requires", 0.8),
        ("Student's unpaired t-test", "Independence of observations", "requires", 1.0),
        ("Student's paired t-test", "Independence of observations", "requires", 0.5),
        ("Chi-square test", "Expected cell frequency >= 5", "requires", 1.0),
        ("Chi-square test", "Independence of observations", "requires", 1.0),
        # Non-parametric alternatives
        ("Mann-Whitney U test", "Student's unpaired t-test", "is_alternative_to", 1.0),
        ("Mann-Whitney U test", "Normality assumption", "does_not_require", 1.0),
        ("Kruskal-Wallis test", "One-way ANOVA", "is_alternative_to", 1.0),
        ("Kruskal-Wallis test", "Normality assumption", "does_not_require", 1.0),
        ("Fisher's exact test", "Chi-square test", "is_alternative_to", 1.0),
        # When to prefer alternatives
        ("Fisher's exact test", "Sparse contingency table", "preferred_when", 1.0),
        ("Mann-Whitney U test", "Non-normal distribution", "preferred_when", 1.0),
        ("Mann-Whitney U test", "Small sample (N<30)", "preferred_when", 0.9),
        ("Kruskal-Wallis test", "Non-normal distribution", "preferred_when", 1.0),
        ("Shapiro-Wilk test", "Normality assumption", "tests", 1.0),
        ("Shapiro-Wilk test", "Numerical variable", "applies_to", 1.0),
        # Study design applicability
        ("Student's unpaired t-test", "Two-group comparison", "applies_to", 1.0),
        ("Mann-Whitney U test", "Two-group comparison", "applies_to", 1.0),
        ("One-way ANOVA", "Multi-group comparison", "applies_to", 1.0),
        ("Kruskal-Wallis test", "Multi-group comparison", "applies_to", 1.0),
        ("Z-test of proportions", "Single cohort study", "applies_to", 1.0),
        ("Chi-square test", "Categorical variable", "applies_to", 1.0),
        ("Fisher's exact test", "Categorical variable", "applies_to", 1.0),
        ("Student's unpaired t-test", "Numerical variable", "applies_to", 1.0),
        ("Mann-Whitney U test", "Numerical variable", "applies_to", 1.0),
        ("Mann-Whitney U test", "Ordinal variable", "applies_to", 1.0),
        # ICMR recommendations
        ("ICMR Statistical Guidelines 2022", "Shapiro-Wilk test", "recommends", 1.0),
        ("ICMR Statistical Guidelines 2022", "Fisher's exact test", "recommends_when", 1.0),
        ("ICMR Statistical Guidelines 2022", "Sparse contingency table", "recommends_when", 1.0),
        ("ICMR Statistical Guidelines 2022", "Mann-Whitney U test", "recommends_when", 0.9),
        ("ICMR Statistical Guidelines 2022", "Small sample (N<30)", "recommends_when", 0.8),
        # p-value and multiple testing
        ("p-value", "Type I error", "related_to", 1.0),
        ("Bonferroni correction", "p-value", "adjusts", 1.0),
        ("Bonferroni correction", "Type I error", "controls", 1.0),
        ("Statistical power", "Type II error", "related_to", 1.0),
        ("Small sample (N<30)", "Statistical power", "reduces", 0.9),
        ("Small sample (N<20)", "Statistical power", "reduces", 1.0),
        # Wrong test combinations (for review flags)
        ("Student's unpaired t-test", "Categorical variable", "inappropriate_for", 1.0),
        ("Chi-square test", "Numerical variable", "inappropriate_for", 0.9),
        ("Z-test of proportions", "Numerical variable", "inappropriate_for", 1.0),
        ("One-way ANOVA", "Two-group comparison", "suboptimal_for", 0.7),
        ("Student's unpaired t-test", "Multi-group comparison", "inappropriate_for", 1.0),
    ]

    # Expand with cross-links between all parametric/non-parametric pairs
    parametric = [
        "Student's unpaired t-test", "Student's paired t-test", "One-way ANOVA", "Chi-square test"
    ]
    non_parametric = [
        "Mann-Whitney U test", "Kruskal-Wallis test", "Fisher's exact test"
    ]
    for p in parametric:
        for np in non_parametric:
            if (p, np, "is_alternative_to", 0.5) not in [(a, b, r, w) for a, b, r, w in edges]:
                if p == "Chi-square test" and np == "Fisher's exact test":
                    continue
                if p == "Student's unpaired t-test" and np == "Mann-Whitney U test":
                    continue
                if p == "One-way ANOVA" and np == "Kruskal-Wallis test":
                    continue

    # Guideline links
    for g in ["CONSORT 2010", "STROBE"]:
        edges.append((g, "Two-group comparison", "applies_to", 0.8))
        edges.append((g, "Single cohort study", "applies_to", 0.8))

    # Concept links to tests
    for test in [
        "Chi-square test", "Fisher's exact test", "Student's unpaired t-test",
        "Mann-Whitney U test", "One-way ANOVA", "Kruskal-Wallis test",
    ]:
        edges.append((test, "p-value", "produces", 1.0))
        edges.append((test, "Confidence interval", "may_report", 0.7))

    result = [{"from_name": a, "to_name": b, "relation": r, "weight": w} for a, b, r, w in edges]
    return result
