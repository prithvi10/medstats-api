"""Unit tests for the deterministic statistical engine."""
import pytest
import numpy as np
import pandas as pd
from src.medstats.core.stats_engine import StatsEngine, TestType


@pytest.fixture
def engine():
    return StatsEngine()


@pytest.fixture
def categorical_series():
    """Simulates MEDICINE 5 mechanism of injury distribution."""
    data = (["RTA"] * 21 + ["Simple fall"] * 7 + ["Fall from height"] * 7 +
            ["Sports related"] * 5 + ["Assault"] * 2)
    return pd.Series(data, name="Mechanism of Injury")


@pytest.fixture
def numerical_series():
    """Simulates MEDICINE 5 operative time (minutes)."""
    np.random.seed(42)
    return pd.Series(
        np.random.normal(53, 12, 50).clip(35, 90).astype(int),
        name="Operative Time"
    )


@pytest.fixture
def two_groups():
    """Simulates two-group comparison (Group L vs Group B heart rate)."""
    np.random.seed(42)
    group1 = pd.Series(np.random.normal(76, 8, 63), name="HR_Group_L")
    group2 = pd.Series(np.random.normal(78, 7, 63), name="HR_Group_B")
    return group1, group2


class TestDescriptive:
    def test_numerical_descriptive(self, engine, numerical_series):
        result = engine.compute_descriptive(numerical_series, "Operative Time")
        assert result.n == 50
        assert result.mean is not None
        assert result.std is not None
        assert result.min_val is not None
        assert result.max_val is not None
        assert 35 <= result.mean <= 90

    def test_categorical_descriptive(self, engine, categorical_series):
        result = engine.compute_descriptive(categorical_series, "Mechanism of Injury")
        assert result.n == 42
        assert result.frequencies is not None
        assert result.percentages is not None
        assert "RTA" in result.frequencies
        assert abs(result.percentages["RTA"] - 50.0) < 1.0

    def test_frequency_table(self, engine, categorical_series):
        table = engine.compute_frequency_table(categorical_series, "Mechanism of Injury")
        assert table["variable_name"] == "Mechanism of Injury"
        assert len(table["rows"]) > 0
        total_row = [r for r in table["rows"] if r["category"] == "TOTAL"]
        assert len(total_row) == 1
        assert total_row[0]["count"] == 42
        assert total_row[0]["percentage"] == 100.0


class TestZTestProportion:
    def test_significant_dominant_category(self, engine, categorical_series):
        result = engine.z_test_proportion(categorical_series, "Mechanism of Injury")
        assert result.test_type == TestType.Z_TEST_PROPORTION
        assert result.statistic is not None
        assert result.p_value is not None
        assert result.details["dominant_category"] == "RTA"
        # RTA is 50% vs expected 20% (5 categories), should be significant
        assert result.is_significant_05

    def test_uniform_distribution_not_significant(self, engine):
        uniform = pd.Series(["A"] * 25 + ["B"] * 25 + ["C"] * 25 + ["D"] * 25)
        result = engine.z_test_proportion(uniform, "Uniform Var")
        assert not result.is_significant_05

    def test_single_category(self, engine):
        single = pd.Series(["Only"] * 50)
        result = engine.z_test_proportion(single, "Single")
        assert "Insufficient" in result.significance_statement


class TestChiSquare:
    def test_independent_variables(self, engine):
        np.random.seed(42)
        gender = pd.Series(["M"] * 35 + ["F"] * 15)
        injury = pd.Series(np.random.choice(["RTA", "Fall", "Sports"], 50))
        result = engine.chi_square_test(gender, injury, "Gender vs Injury")
        assert result.test_type in (TestType.CHI_SQUARE, TestType.FISHER_EXACT)
        assert result.p_value is not None

    def test_associated_variables(self, engine):
        # Strongly associated: males mostly RTA, females mostly Fall
        gender = pd.Series(["M"] * 40 + ["F"] * 40)
        injury = pd.Series(["RTA"] * 35 + ["Fall"] * 5 + ["Fall"] * 35 + ["RTA"] * 5)
        result = engine.chi_square_test(gender, injury, "Gender vs Injury")
        assert result.is_significant_01


class TestTTest:
    def test_similar_groups_not_significant(self, engine, two_groups):
        g1, g2 = two_groups
        result = engine.unpaired_t_test(g1, g2, "Heart Rate")
        assert result.test_type == TestType.UNPAIRED_T_TEST
        assert result.p_value is not None
        assert result.details["group1_n"] == 63
        assert result.details["group2_n"] == 63

    def test_different_groups_significant(self, engine):
        np.random.seed(42)
        g1 = pd.Series(np.random.normal(70, 5, 50))
        g2 = pd.Series(np.random.normal(85, 5, 50))
        result = engine.unpaired_t_test(g1, g2, "BP")
        assert result.is_significant_01

    def test_paired_t_test(self, engine):
        np.random.seed(42)
        before = pd.Series(np.random.normal(120, 10, 30))
        after = pd.Series(before - np.random.normal(15, 5, 30))
        result = engine.paired_t_test(before, after, "SBP")
        assert result.test_type == TestType.PAIRED_T_TEST
        assert result.is_significant_05


class TestMannWhitneyU:
    def test_non_parametric_comparison(self, engine):
        np.random.seed(42)
        g1 = pd.Series(np.random.exponential(5, 30))
        g2 = pd.Series(np.random.exponential(10, 30))
        result = engine.mann_whitney_u(g1, g2, "Hospital Stay")
        assert result.test_type == TestType.MANN_WHITNEY_U
        assert result.p_value is not None


class TestANOVA:
    def test_one_way_anova_significant(self, engine):
        np.random.seed(42)
        groups = [
            pd.Series(np.random.normal(50, 5, 20)),
            pd.Series(np.random.normal(60, 5, 20)),
            pd.Series(np.random.normal(70, 5, 20)),
        ]
        result = engine.one_way_anova(groups, "Operative Time by Fracture Type")
        assert result.test_type == TestType.ONE_WAY_ANOVA
        assert result.is_significant_01

    def test_one_way_anova_not_significant(self, engine):
        np.random.seed(42)
        groups = [
            pd.Series(np.random.normal(50, 5, 20)),
            pd.Series(np.random.normal(51, 5, 20)),
            pd.Series(np.random.normal(50.5, 5, 20)),
        ]
        result = engine.one_way_anova(groups, "Similar Groups")
        assert not result.is_significant_05

    def test_kruskal_wallis(self, engine):
        np.random.seed(42)
        groups = [
            pd.Series(np.random.exponential(3, 15)),
            pd.Series(np.random.exponential(8, 15)),
            pd.Series(np.random.exponential(15, 15)),
        ]
        result = engine.kruskal_wallis(groups, "Union Time")
        assert result.test_type == TestType.KRUSKAL_WALLIS
        assert result.p_value is not None


class TestNormality:
    def test_normal_data(self, engine):
        np.random.seed(42)
        normal = pd.Series(np.random.normal(0, 1, 100))
        result = engine.shapiro_wilk(normal, "Normal Var")
        assert result.details["is_normal"]

    def test_non_normal_data(self, engine):
        np.random.seed(42)
        skewed = pd.Series(np.random.exponential(2, 100))
        result = engine.shapiro_wilk(skewed, "Skewed Var")
        assert not result.details["is_normal"]

    def test_too_small_sample(self, engine):
        tiny = pd.Series([1.0, 2.0])
        result = engine.shapiro_wilk(tiny, "Tiny")
        assert "outside valid range" in result.significance_statement


class TestResultSerialization:
    def test_to_dict(self, engine, categorical_series):
        result = engine.z_test_proportion(categorical_series, "Test Var")
        d = result.to_dict()
        assert isinstance(d, dict)
        assert d["test_type"] == "z_test_proportion"
        assert "p_value" in d
        assert "significance_statement" in d
