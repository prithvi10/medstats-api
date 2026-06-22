"""
Statistical Engine - Deterministic computation of statistical tests.
NO LLM involvement. Pure scipy/statsmodels/pingouin.
"""
import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum


class TestType(str, Enum):
    DESCRIPTIVE = "descriptive"
    Z_TEST_PROPORTION = "z_test_proportion"
    CHI_SQUARE = "chi_square"
    FISHER_EXACT = "fisher_exact"
    UNPAIRED_T_TEST = "unpaired_t_test"
    PAIRED_T_TEST = "paired_t_test"
    MANN_WHITNEY_U = "mann_whitney_u"
    WILCOXON = "wilcoxon"
    ONE_WAY_ANOVA = "one_way_anova"
    KRUSKAL_WALLIS = "kruskal_wallis"
    PEARSON_CORRELATION = "pearson_correlation"
    SPEARMAN_CORRELATION = "spearman_correlation"
    SHAPIRO_WILK = "shapiro_wilk"


@dataclass
class DescriptiveResult:
    variable_name: str
    n: int
    mean: Optional[float] = None
    std: Optional[float] = None
    median: Optional[float] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    frequencies: Optional[Dict[str, int]] = None
    percentages: Optional[Dict[str, float]] = None


@dataclass
class TestResult:
    test_type: TestType
    variable_name: str
    statistic: Optional[float] = None
    p_value: Optional[float] = None
    is_significant_05: bool = False
    is_significant_01: bool = False
    significance_statement: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    descriptive: Optional[DescriptiveResult] = None

    def to_dict(self) -> dict:
        return {
            "test_type": self.test_type.value,
            "variable_name": self.variable_name,
            "statistic": self.statistic,
            "p_value": self.p_value,
            "is_significant_05": self.is_significant_05,
            "is_significant_01": self.is_significant_01,
            "significance_statement": self.significance_statement,
            "details": self.details,
            "descriptive": vars(self.descriptive) if self.descriptive else None,
        }


class StatsEngine:
    """Deterministic statistical computation engine."""

    def compute_descriptive(self, series: pd.Series, variable_name: str) -> DescriptiveResult:
        """Compute descriptive statistics for a single variable."""
        if pd.api.types.is_numeric_dtype(series):
            clean = series.dropna()
            return DescriptiveResult(
                variable_name=variable_name,
                n=len(clean),
                mean=round(float(clean.mean()), 2),
                std=round(float(clean.std()), 2),
                median=round(float(clean.median()), 2),
                min_val=round(float(clean.min()), 2),
                max_val=round(float(clean.max()), 2),
            )
        else:
            freq = series.value_counts()
            total = freq.sum()
            return DescriptiveResult(
                variable_name=variable_name,
                n=int(total),
                frequencies=freq.to_dict(),
                percentages={k: round(v / total * 100, 2) for k, v in freq.items()},
            )

    def compute_frequency_table(self, series: pd.Series, variable_name: str) -> Dict[str, Any]:
        """Generate frequency table with counts and percentages."""
        freq = series.value_counts()
        total = freq.sum()
        table = []
        for category, count in freq.items():
            table.append({
                "category": str(category),
                "count": int(count),
                "percentage": round(count / total * 100, 2),
            })
        table.append({"category": "TOTAL", "count": int(total), "percentage": 100.0})
        return {"variable_name": variable_name, "rows": table, "total": int(total)}

    def z_test_proportion(self, series: pd.Series, variable_name: str) -> TestResult:
        """Z-test of proportions: tests if one category significantly dominates."""
        freq = series.value_counts()
        if len(freq) < 2:
            return TestResult(
                test_type=TestType.Z_TEST_PROPORTION,
                variable_name=variable_name,
                significance_statement="Insufficient categories for comparison.",
            )

        n = freq.sum()
        max_category = freq.index[0]
        max_count = freq.iloc[0]
        p_observed = max_count / n
        p_null = 1 / len(freq)

        se = np.sqrt(p_null * (1 - p_null) / n)
        if se == 0:
            z_stat = 0.0
        else:
            z_stat = (p_observed - p_null) / se

        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

        result = TestResult(
            test_type=TestType.Z_TEST_PROPORTION,
            variable_name=variable_name,
            statistic=round(z_stat, 4),
            p_value=round(p_value, 4),
            is_significant_05=p_value < 0.05,
            is_significant_01=p_value < 0.01,
            details={"dominant_category": str(max_category), "proportion": round(p_observed, 4)},
        )
        result.significance_statement = self._format_significance(
            f"the proportion of {max_category}", result
        )
        return result

    def chi_square_test(
        self, series1: pd.Series, series2: pd.Series, var_name: str
    ) -> TestResult:
        """Chi-square test of independence between two categorical variables."""
        contingency = pd.crosstab(series1, series2)

        if contingency.size < 4:
            return self.fisher_exact_test(series1, series2, var_name)

        min_expected = stats.contingency.expected_freq(contingency.values).min()
        if min_expected < 5:
            return self.fisher_exact_test(series1, series2, var_name)

        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

        result = TestResult(
            test_type=TestType.CHI_SQUARE,
            variable_name=var_name,
            statistic=round(chi2, 4),
            p_value=round(p_value, 4),
            is_significant_05=p_value < 0.05,
            is_significant_01=p_value < 0.01,
            details={"dof": dof, "contingency_table": contingency.to_dict()},
        )
        result.significance_statement = self._format_chi_square(result)
        return result

    def fisher_exact_test(
        self, series1: pd.Series, series2: pd.Series, var_name: str
    ) -> TestResult:
        """Fisher's exact test for 2x2 tables or small expected frequencies."""
        contingency = pd.crosstab(series1, series2)

        if contingency.shape == (2, 2):
            odds_ratio, p_value = stats.fisher_exact(contingency.values)
        else:
            chi2, p_value, dof, _ = stats.chi2_contingency(contingency)
            odds_ratio = None

        result = TestResult(
            test_type=TestType.FISHER_EXACT,
            variable_name=var_name,
            p_value=round(p_value, 4),
            is_significant_05=p_value < 0.05,
            is_significant_01=p_value < 0.01,
            details={"odds_ratio": odds_ratio},
        )
        result.significance_statement = self._format_significance(
            f"association in {var_name}", result
        )
        return result

    def unpaired_t_test(
        self, group1: pd.Series, group2: pd.Series, var_name: str
    ) -> TestResult:
        """Student's unpaired t-test for comparing means of two groups."""
        g1 = group1.dropna().astype(float)
        g2 = group2.dropna().astype(float)

        t_stat, p_value = stats.ttest_ind(g1, g2)

        result = TestResult(
            test_type=TestType.UNPAIRED_T_TEST,
            variable_name=var_name,
            statistic=round(t_stat, 4),
            p_value=round(p_value, 4),
            is_significant_05=p_value < 0.05,
            is_significant_01=p_value < 0.01,
            details={
                "group1_mean": round(float(g1.mean()), 2),
                "group1_std": round(float(g1.std()), 2),
                "group2_mean": round(float(g2.mean()), 2),
                "group2_std": round(float(g2.std()), 2),
                "group1_n": len(g1),
                "group2_n": len(g2),
            },
        )
        result.significance_statement = self._format_t_test(result)
        return result

    def paired_t_test(
        self, before: pd.Series, after: pd.Series, var_name: str
    ) -> TestResult:
        """Paired t-test for before/after comparisons."""
        b = before.dropna().astype(float)
        a = after.dropna().astype(float)
        min_len = min(len(b), len(a))

        t_stat, p_value = stats.ttest_rel(b[:min_len], a[:min_len])

        result = TestResult(
            test_type=TestType.PAIRED_T_TEST,
            variable_name=var_name,
            statistic=round(t_stat, 4),
            p_value=round(p_value, 4),
            is_significant_05=p_value < 0.05,
            is_significant_01=p_value < 0.01,
        )
        result.significance_statement = self._format_significance(
            f"difference in {var_name}", result
        )
        return result

    def mann_whitney_u(
        self, group1: pd.Series, group2: pd.Series, var_name: str
    ) -> TestResult:
        """Mann-Whitney U test (non-parametric alternative to unpaired t-test)."""
        g1 = group1.dropna().astype(float)
        g2 = group2.dropna().astype(float)

        u_stat, p_value = stats.mannwhitneyu(g1, g2, alternative="two-sided")

        result = TestResult(
            test_type=TestType.MANN_WHITNEY_U,
            variable_name=var_name,
            statistic=round(u_stat, 4),
            p_value=round(p_value, 4),
            is_significant_05=p_value < 0.05,
            is_significant_01=p_value < 0.01,
        )
        result.significance_statement = self._format_significance(
            f"difference in {var_name}", result
        )
        return result

    def one_way_anova(self, groups: List[pd.Series], var_name: str) -> TestResult:
        """One-way ANOVA for comparing means across 3+ groups."""
        clean_groups = [g.dropna().astype(float) for g in groups]
        clean_groups = [g for g in clean_groups if len(g) > 0]

        if len(clean_groups) < 2:
            return TestResult(
                test_type=TestType.ONE_WAY_ANOVA,
                variable_name=var_name,
                significance_statement="Insufficient groups for ANOVA.",
            )

        f_stat, p_value = stats.f_oneway(*clean_groups)

        result = TestResult(
            test_type=TestType.ONE_WAY_ANOVA,
            variable_name=var_name,
            statistic=round(f_stat, 4),
            p_value=round(p_value, 4),
            is_significant_05=p_value < 0.05,
            is_significant_01=p_value < 0.01,
            details={"num_groups": len(clean_groups)},
        )
        result.significance_statement = self._format_significance(
            f"difference in {var_name} across groups", result
        )
        return result

    def kruskal_wallis(self, groups: List[pd.Series], var_name: str) -> TestResult:
        """Kruskal-Wallis H test (non-parametric ANOVA)."""
        clean_groups = [g.dropna().astype(float) for g in groups]
        clean_groups = [g for g in clean_groups if len(g) > 0]

        if len(clean_groups) < 2:
            return TestResult(
                test_type=TestType.KRUSKAL_WALLIS,
                variable_name=var_name,
                significance_statement="Insufficient groups.",
            )

        h_stat, p_value = stats.kruskal(*clean_groups)

        result = TestResult(
            test_type=TestType.KRUSKAL_WALLIS,
            variable_name=var_name,
            statistic=round(h_stat, 4),
            p_value=round(p_value, 4),
            is_significant_05=p_value < 0.05,
            is_significant_01=p_value < 0.01,
        )
        result.significance_statement = self._format_significance(
            f"difference in {var_name}", result
        )
        return result

    def shapiro_wilk(self, series: pd.Series, var_name: str) -> TestResult:
        """Shapiro-Wilk normality test."""
        clean = series.dropna().astype(float)
        if len(clean) < 3 or len(clean) > 5000:
            return TestResult(
                test_type=TestType.SHAPIRO_WILK,
                variable_name=var_name,
                significance_statement="Sample size outside valid range for Shapiro-Wilk (3-5000).",
            )

        w_stat, p_value = stats.shapiro(clean)

        is_normal = p_value > 0.05
        result = TestResult(
            test_type=TestType.SHAPIRO_WILK,
            variable_name=var_name,
            statistic=round(w_stat, 4),
            p_value=round(p_value, 4),
            is_significant_05=p_value < 0.05,
            details={"is_normal": is_normal},
        )
        result.significance_statement = (
            f"Data for {var_name} is {'normally' if is_normal else 'not normally'} distributed "
            f"(W={w_stat:.4f}, p={p_value:.4f})."
        )
        return result

    def _format_significance(self, description: str, result: TestResult) -> str:
        if result.p_value is None:
            return description
        if result.is_significant_01:
            return f"By applying the test, {description} is highly significant (p<0.01)."
        elif result.is_significant_05:
            return f"By applying the test, {description} is significant (p<0.05)."
        else:
            return f"p={result.p_value:.4f}, not significant."

    def _format_chi_square(self, result: TestResult) -> str:
        chi2 = result.statistic
        p = result.p_value
        if result.is_significant_01:
            return f"Value of X²={chi2:.3f}, p={p:.4f}, highly significant."
        elif result.is_significant_05:
            return f"Value of X²={chi2:.3f}, p={p:.4f}, significant."
        else:
            return f"Value of X²={chi2:.3f}, p={p:.4f}, not significant."

    def _format_t_test(self, result: TestResult) -> str:
        t = result.statistic
        p = result.p_value
        d = result.details
        if result.is_significant_05:
            sig = "significant"
        else:
            sig = "not significant"
        return (
            f"By applying Student's Unpaired t-test, the difference in mean values "
            f"({d.get('group1_mean', '')} vs {d.get('group2_mean', '')}) is {sig} "
            f"(t={t:.3f}, p={p:.4f})."
        )
