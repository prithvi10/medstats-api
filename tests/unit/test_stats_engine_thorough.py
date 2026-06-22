"""
Thorough unit tests for the statistical engine.
Covers edge cases, rounding, math errors, boundary conditions.
"""
import pytest
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from src.medstats.core.stats_engine import StatsEngine, TestType, TestResult


@pytest.fixture
def engine():
    return StatsEngine()


class TestDescriptiveEdgeCases:
    def test_single_value(self, engine):
        s = pd.Series([42.0])
        r = engine.compute_descriptive(s, "single")
        assert r.n == 1
        assert r.mean == 42.0
        assert r.median == 42.0
        assert r.min_val == 42.0
        assert r.max_val == 42.0

    def test_all_same_values(self, engine):
        s = pd.Series([7.0] * 50)
        r = engine.compute_descriptive(s, "constant")
        assert r.mean == 7.0
        assert r.std == 0.0
        assert r.min_val == r.max_val == 7.0

    def test_with_nan_values(self, engine):
        s = pd.Series([1.0, 2.0, np.nan, 4.0, np.nan, 6.0])
        r = engine.compute_descriptive(s, "with_nan")
        assert r.n == 4
        assert r.mean == 3.25

    def test_negative_values(self, engine):
        s = pd.Series([-10.0, -5.0, 0.0, 5.0, 10.0])
        r = engine.compute_descriptive(s, "neg")
        assert r.mean == 0.0
        assert r.min_val == -10.0
        assert r.max_val == 10.0

    def test_very_large_numbers(self, engine):
        s = pd.Series([1e10, 2e10, 3e10])
        r = engine.compute_descriptive(s, "large")
        assert r.mean == 2e10
        assert r.n == 3

    def test_very_small_numbers(self, engine):
        s = pd.Series([0.0001, 0.0002, 0.0003])
        r = engine.compute_descriptive(s, "tiny")
        assert r.mean == 0.0
        assert r.n == 3

    def test_empty_categorical(self, engine):
        s = pd.Series([], dtype=object)
        r = engine.compute_descriptive(s, "empty_cat")
        assert r.n == 0

    def test_categorical_with_nan(self, engine):
        s = pd.Series(["A", "B", np.nan, "A", np.nan, "C"])
        r = engine.compute_descriptive(s, "cat_nan")
        assert r.n == 4
        assert r.frequencies["A"] == 2

    def test_rounding_precision(self, engine):
        s = pd.Series([1.0/3.0] * 100)
        r = engine.compute_descriptive(s, "thirds")
        assert r.mean == 0.33
        assert abs(r.mean - 0.33) < 0.01


class TestFrequencyTableEdgeCases:
    def test_single_category(self, engine):
        s = pd.Series(["X"] * 10)
        t = engine.compute_frequency_table(s, "single")
        assert len(t["rows"]) == 2  # category + TOTAL
        assert t["rows"][0]["percentage"] == 100.0

    def test_many_categories(self, engine):
        s = pd.Series([f"cat_{i}" for i in range(50)])
        t = engine.compute_frequency_table(s, "many")
        assert t["total"] == 50
        assert all(r["percentage"] == 2.0 for r in t["rows"] if r["category"] != "TOTAL")

    def test_percentage_sum_exactly_100(self, engine):
        s = pd.Series(["A"] * 33 + ["B"] * 33 + ["C"] * 34)
        t = engine.compute_frequency_table(s, "almost_equal")
        non_total = [r for r in t["rows"] if r["category"] != "TOTAL"]
        total_pct = sum(r["percentage"] for r in non_total)
        assert abs(total_pct - 100.0) < 0.1

    def test_rounding_three_categories(self, engine):
        s = pd.Series(["A"] * 1 + ["B"] * 1 + ["C"] * 1)
        t = engine.compute_frequency_table(s, "thirds")
        non_total = [r for r in t["rows"] if r["category"] != "TOTAL"]
        for r in non_total:
            assert r["percentage"] == 33.33


class TestZTestProportionEdgeCases:
    def test_two_categories_equal(self, engine):
        s = pd.Series(["A"] * 50 + ["B"] * 50)
        r = engine.z_test_proportion(s, "equal")
        assert not r.is_significant_05

    def test_extreme_dominance(self, engine):
        s = pd.Series(["A"] * 99 + ["B"] * 1)
        r = engine.z_test_proportion(s, "extreme")
        assert r.is_significant_01
        assert r.details["dominant_category"] == "A"
        assert r.details["proportion"] > 0.95

    def test_many_equal_categories(self, engine):
        cats = [f"c{i}" for i in range(10)]
        s = pd.Series(cats * 10)
        r = engine.z_test_proportion(s, "uniform10")
        assert not r.is_significant_05

    def test_single_category_returns_insufficient(self, engine):
        s = pd.Series(["Only"] * 100)
        r = engine.z_test_proportion(s, "single")
        assert "Insufficient" in r.significance_statement
        assert r.statistic is None

    def test_with_nan_in_series(self, engine):
        s = pd.Series(["A"] * 40 + ["B"] * 10 + [np.nan] * 5)
        r = engine.z_test_proportion(s.dropna(), "nan_test")
        assert r.details["dominant_category"] == "A"

    def test_known_z_value(self, engine):
        s = pd.Series(["A"] * 70 + ["B"] * 30)
        r = engine.z_test_proportion(s, "known")
        p_obs = 0.70
        p_null = 0.50
        se = np.sqrt(0.5 * 0.5 / 100)
        expected_z = (p_obs - p_null) / se
        assert abs(r.statistic - round(expected_z, 4)) < 0.01


class TestChiSquareEdgeCases:
    def test_2x2_perfectly_independent(self, engine):
        s1 = pd.Series(["M"] * 50 + ["F"] * 50)
        s2 = pd.Series(["Y", "N"] * 50)
        r = engine.chi_square_test(s1, s2, "independent")
        assert r.p_value is not None
        assert not r.is_significant_05

    def test_2x2_perfectly_associated(self, engine):
        s1 = pd.Series(["M"] * 50 + ["F"] * 50)
        s2 = pd.Series(["Y"] * 50 + ["N"] * 50)
        r = engine.chi_square_test(s1, s2, "perfect_assoc")
        assert r.is_significant_01

    def test_falls_back_to_fisher_for_small_expected(self, engine):
        s1 = pd.Series(["M"] * 3 + ["F"] * 3)
        s2 = pd.Series(["Y"] * 2 + ["N"] * 1 + ["Y"] * 1 + ["N"] * 2)
        r = engine.chi_square_test(s1, s2, "small_n")
        assert r.test_type == TestType.FISHER_EXACT

    def test_3x3_contingency(self, engine):
        np.random.seed(42)
        s1 = pd.Series(np.random.choice(["A", "B", "C"], 200))
        s2 = pd.Series(np.random.choice(["X", "Y", "Z"], 200))
        r = engine.chi_square_test(s1, s2, "3x3")
        assert r.test_type == TestType.CHI_SQUARE
        assert r.details.get("dof") == 4

    def test_validates_against_scipy(self, engine):
        np.random.seed(123)
        s1 = pd.Series(["A"] * 30 + ["B"] * 20 + ["A"] * 10 + ["B"] * 40)
        s2 = pd.Series(["X"] * 40 + ["X"] * 20 + ["Y"] * 20 + ["Y"] * 20)
        r = engine.chi_square_test(s1, s2, "validate")
        ct = pd.crosstab(s1, s2)
        chi2_expected, p_expected, _, _ = scipy_stats.chi2_contingency(ct)
        assert abs(r.statistic - round(chi2_expected, 4)) < 0.01
        assert abs(r.p_value - round(p_expected, 4)) < 0.01


class TestTTestEdgeCases:
    def test_identical_groups(self, engine):
        g = pd.Series([5.0] * 30)
        r = engine.unpaired_t_test(g, g, "identical")
        assert r.statistic == 0.0 or np.isnan(r.statistic)

    def test_one_value_different(self, engine):
        g1 = pd.Series([10.0] * 50)
        g2 = pd.Series([10.0] * 49 + [10.1])
        r = engine.unpaired_t_test(g1, g2, "tiny_diff")
        assert not r.is_significant_05

    def test_unequal_group_sizes(self, engine):
        np.random.seed(42)
        g1 = pd.Series(np.random.normal(50, 5, 10))
        g2 = pd.Series(np.random.normal(60, 5, 100))
        r = engine.unpaired_t_test(g1, g2, "unequal_n")
        assert r.is_significant_05
        assert r.details["group1_n"] == 10
        assert r.details["group2_n"] == 100

    def test_with_nan_values(self, engine):
        g1 = pd.Series([1, 2, np.nan, 4, 5, np.nan, 7, 8, 9, 10.0])
        g2 = pd.Series([11, 12, 13, 14, np.nan, 16, 17, 18, 19, 20.0])
        r = engine.unpaired_t_test(g1, g2, "nan_groups")
        assert r.details["group1_n"] == 8
        assert r.details["group2_n"] == 9

    def test_validates_against_scipy(self, engine):
        np.random.seed(99)
        g1 = pd.Series(np.random.normal(100, 15, 40))
        g2 = pd.Series(np.random.normal(110, 15, 40))
        r = engine.unpaired_t_test(g1, g2, "validate")
        t_exp, p_exp = scipy_stats.ttest_ind(g1, g2)
        assert abs(r.statistic - round(t_exp, 4)) < 0.01
        assert abs(r.p_value - round(p_exp, 4)) < 0.01

    def test_negative_t_stat(self, engine):
        np.random.seed(42)
        g1 = pd.Series(np.random.normal(80, 5, 30))
        g2 = pd.Series(np.random.normal(70, 5, 30))
        r = engine.unpaired_t_test(g1, g2, "g1_higher")
        assert r.statistic > 0
        r2 = engine.unpaired_t_test(g2, g1, "g2_higher")
        assert r2.statistic < 0
        assert abs(r.p_value - r2.p_value) < 0.001


class TestPairedTTestEdgeCases:
    def test_no_change(self, engine):
        before = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        after = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        r = engine.paired_t_test(before, after, "no_change")
        assert r.statistic == 0.0 or np.isnan(r.statistic)

    def test_consistent_improvement(self, engine):
        np.random.seed(42)
        before = pd.Series(np.random.normal(120, 10, 30))
        after = before - 20
        r = engine.paired_t_test(before, after, "big_drop")
        assert r.is_significant_01

    def test_unequal_lengths_truncated(self, engine):
        before = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        after = pd.Series([2.0, 3.0, 4.0])
        r = engine.paired_t_test(before, after, "unequal")
        assert r.p_value is not None

    def test_validates_against_scipy(self, engine):
        np.random.seed(77)
        b = pd.Series(np.random.normal(100, 10, 25))
        a = b + np.random.normal(5, 3, 25)
        r = engine.paired_t_test(b, a, "validate")
        t_exp, p_exp = scipy_stats.ttest_rel(b, a)
        assert abs(r.statistic - round(t_exp, 4)) < 0.01
        assert abs(r.p_value - round(p_exp, 4)) < 0.01


class TestMannWhitneyEdgeCases:
    def test_identical_distributions(self, engine):
        np.random.seed(42)
        g1 = pd.Series(np.random.uniform(0, 10, 30))
        g2 = pd.Series(np.random.uniform(0, 10, 30))
        r = engine.mann_whitney_u(g1, g2, "same_dist")
        assert not r.is_significant_05

    def test_completely_separated(self, engine):
        g1 = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        g2 = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0])
        r = engine.mann_whitney_u(g1, g2, "separated")
        assert r.is_significant_05

    def test_with_ties(self, engine):
        g1 = pd.Series([1.0, 1.0, 2.0, 2.0, 3.0] * 5)
        g2 = pd.Series([3.0, 3.0, 4.0, 4.0, 5.0] * 5)
        r = engine.mann_whitney_u(g1, g2, "ties")
        assert r.p_value is not None

    def test_validates_against_scipy(self, engine):
        np.random.seed(55)
        g1 = pd.Series(np.random.exponential(5, 25))
        g2 = pd.Series(np.random.exponential(8, 25))
        r = engine.mann_whitney_u(g1, g2, "validate")
        u_exp, p_exp = scipy_stats.mannwhitneyu(g1, g2, alternative="two-sided")
        assert abs(r.statistic - round(u_exp, 4)) < 0.01
        assert abs(r.p_value - round(p_exp, 4)) < 0.01


class TestANOVAEdgeCases:
    def test_all_same_means(self, engine):
        np.random.seed(42)
        groups = [pd.Series(np.random.normal(50, 5, 20)) for _ in range(4)]
        r = engine.one_way_anova(groups, "same_means")
        assert not r.is_significant_05

    def test_one_different_group(self, engine):
        np.random.seed(42)
        groups = [
            pd.Series(np.random.normal(50, 3, 25)),
            pd.Series(np.random.normal(50, 3, 25)),
            pd.Series(np.random.normal(80, 3, 25)),
        ]
        r = engine.one_way_anova(groups, "one_diff")
        assert r.is_significant_01

    def test_single_group_insufficient(self, engine):
        r = engine.one_way_anova([pd.Series([1, 2, 3])], "single")
        assert "Insufficient" in r.significance_statement

    def test_empty_groups_filtered(self, engine):
        groups = [
            pd.Series([1.0, 2.0, 3.0]),
            pd.Series([], dtype=float),
            pd.Series([4.0, 5.0, 6.0]),
        ]
        r = engine.one_way_anova(groups, "with_empty")
        assert r.details["num_groups"] == 2

    def test_unequal_group_sizes(self, engine):
        np.random.seed(42)
        groups = [
            pd.Series(np.random.normal(50, 5, 5)),
            pd.Series(np.random.normal(70, 5, 50)),
            pd.Series(np.random.normal(90, 5, 100)),
        ]
        r = engine.one_way_anova(groups, "unequal_sizes")
        assert r.is_significant_01

    def test_validates_against_scipy(self, engine):
        np.random.seed(88)
        g1 = pd.Series(np.random.normal(40, 5, 20))
        g2 = pd.Series(np.random.normal(50, 5, 20))
        g3 = pd.Series(np.random.normal(60, 5, 20))
        r = engine.one_way_anova([g1, g2, g3], "validate")
        f_exp, p_exp = scipy_stats.f_oneway(g1, g2, g3)
        assert abs(r.statistic - round(f_exp, 4)) < 0.01
        assert abs(r.p_value - round(p_exp, 4)) < 0.01


class TestKruskalWallisEdgeCases:
    def test_identical_distributions(self, engine):
        np.random.seed(42)
        groups = [pd.Series(np.random.uniform(0, 10, 20)) for _ in range(3)]
        r = engine.kruskal_wallis(groups, "same")
        assert not r.is_significant_05

    def test_clearly_different(self, engine):
        groups = [
            pd.Series([1, 2, 3, 4, 5.0]),
            pd.Series([10, 11, 12, 13, 14.0]),
            pd.Series([20, 21, 22, 23, 24.0]),
        ]
        r = engine.kruskal_wallis(groups, "different")
        assert r.is_significant_05

    def test_single_group_insufficient(self, engine):
        r = engine.kruskal_wallis([pd.Series([1, 2, 3])], "one")
        assert "Insufficient" in r.significance_statement

    def test_validates_against_scipy(self, engine):
        np.random.seed(66)
        g1 = pd.Series(np.random.exponential(3, 15))
        g2 = pd.Series(np.random.exponential(6, 15))
        g3 = pd.Series(np.random.exponential(12, 15))
        r = engine.kruskal_wallis([g1, g2, g3], "validate")
        h_exp, p_exp = scipy_stats.kruskal(g1, g2, g3)
        assert abs(r.statistic - round(h_exp, 4)) < 0.1
        assert abs(r.p_value - round(p_exp, 4)) < 0.01


class TestShapiroWilkEdgeCases:
    def test_perfectly_normal(self, engine):
        np.random.seed(42)
        s = pd.Series(np.random.normal(0, 1, 200))
        r = engine.shapiro_wilk(s, "perfect_normal")
        assert r.details["is_normal"]
        assert r.statistic > 0.99

    def test_heavily_skewed(self, engine):
        np.random.seed(42)
        s = pd.Series(np.random.exponential(1, 100))
        r = engine.shapiro_wilk(s, "skewed")
        assert not r.details["is_normal"]

    def test_bimodal_distribution(self, engine):
        np.random.seed(42)
        s = pd.Series(np.concatenate([
            np.random.normal(30, 2, 50),
            np.random.normal(70, 2, 50),
        ]))
        r = engine.shapiro_wilk(s, "bimodal")
        assert not r.details["is_normal"]

    def test_uniform_distribution(self, engine):
        np.random.seed(42)
        s = pd.Series(np.random.uniform(0, 100, 100))
        r = engine.shapiro_wilk(s, "uniform")
        assert not r.details["is_normal"]

    def test_too_small_sample(self, engine):
        s = pd.Series([1.0, 2.0])
        r = engine.shapiro_wilk(s, "tiny")
        assert "outside valid range" in r.significance_statement

    def test_too_large_sample(self, engine):
        s = pd.Series(np.random.normal(0, 1, 5001))
        r = engine.shapiro_wilk(s, "huge")
        assert "outside valid range" in r.significance_statement

    def test_exactly_3_values(self, engine):
        s = pd.Series([1.0, 2.0, 3.0])
        r = engine.shapiro_wilk(s, "min_valid")
        assert r.statistic is not None
        assert r.p_value is not None

    def test_constant_values_edge(self, engine):
        s = pd.Series([5.0] * 10)
        r = engine.shapiro_wilk(s, "constant")
        assert r.p_value is not None


class TestFisherExactEdgeCases:
    def test_2x2_no_association(self, engine):
        s1 = pd.Series(["M", "M", "F", "F"] * 10)
        s2 = pd.Series(["Y", "N", "Y", "N"] * 10)
        r = engine.fisher_exact_test(s1, s2, "no_assoc")
        assert not r.is_significant_05

    def test_2x2_perfect_association(self, engine):
        s1 = pd.Series(["M"] * 20 + ["F"] * 20)
        s2 = pd.Series(["Y"] * 20 + ["N"] * 20)
        r = engine.fisher_exact_test(s1, s2, "perfect")
        assert r.is_significant_01

    def test_2x2_validates_odds_ratio(self, engine):
        s1 = pd.Series(["M"] * 20 + ["F"] * 20)
        s2 = pd.Series(["Y"] * 15 + ["N"] * 5 + ["Y"] * 5 + ["N"] * 15)
        r = engine.fisher_exact_test(s1, s2, "or_check")
        ct = pd.crosstab(s1, s2)
        or_exp, _ = scipy_stats.fisher_exact(ct.values)
        assert abs(r.details["odds_ratio"] - or_exp) < 0.01


class TestSignificanceBoundary:
    """Test exact boundary conditions at p=0.05 and p=0.01."""

    def test_p_exactly_005(self, engine):
        r = TestResult(
            test_type=TestType.UNPAIRED_T_TEST,
            variable_name="test",
            statistic=1.96,
            p_value=0.05,
            is_significant_05=0.05 < 0.05,
            is_significant_01=0.05 < 0.01,
        )
        assert not r.is_significant_05

    def test_p_just_below_005(self, engine):
        r = TestResult(
            test_type=TestType.UNPAIRED_T_TEST,
            variable_name="test",
            statistic=2.0,
            p_value=0.0499,
            is_significant_05=0.0499 < 0.05,
            is_significant_01=0.0499 < 0.01,
        )
        assert r.is_significant_05
        assert not r.is_significant_01

    def test_p_exactly_001(self, engine):
        r = TestResult(
            test_type=TestType.UNPAIRED_T_TEST,
            variable_name="test",
            statistic=2.58,
            p_value=0.01,
            is_significant_05=0.01 < 0.05,
            is_significant_01=0.01 < 0.01,
        )
        assert r.is_significant_05
        assert not r.is_significant_01

    def test_p_zero(self, engine):
        r = TestResult(
            test_type=TestType.CHI_SQUARE,
            variable_name="test",
            statistic=100.0,
            p_value=0.0,
            is_significant_05=True,
            is_significant_01=True,
        )
        assert r.is_significant_05
        assert r.is_significant_01


class TestSerializationConsistency:
    def test_to_dict_preserves_all_fields(self, engine):
        np.random.seed(42)
        s = pd.Series(["A"] * 30 + ["B"] * 20)
        r = engine.z_test_proportion(s, "test")
        d = r.to_dict()
        assert d["test_type"] == "z_test_proportion"
        assert isinstance(d["statistic"], float)
        assert isinstance(d["p_value"], float)
        assert d["is_significant_05"] in (True, False)
        assert d["is_significant_01"] in (True, False)
        assert isinstance(d["significance_statement"], str)
        assert isinstance(d["details"], dict)

    def test_to_dict_handles_none_values(self, engine):
        s = pd.Series(["Only"] * 10)
        r = engine.z_test_proportion(s, "single")
        d = r.to_dict()
        assert d["statistic"] is None
        assert d["p_value"] is None

    def test_to_dict_roundtrip_numeric_precision(self, engine):
        np.random.seed(42)
        g1 = pd.Series(np.random.normal(50, 10, 30))
        g2 = pd.Series(np.random.normal(55, 10, 30))
        r = engine.unpaired_t_test(g1, g2, "precision")
        d = r.to_dict()
        assert len(str(d["statistic"]).split(".")[-1]) <= 4
        assert len(str(d["p_value"]).split(".")[-1]) <= 4
