"""
Review Agent test scenarios for:
1. Wrong test type applied to data → failure case
2. Insignificant results → recommend alternative tests
3. Insufficient data in masterchart → more/better data required
"""
import pytest
import numpy as np
import pandas as pd
from src.medstats.agents.review_agent import ReviewAgent, Severity
from src.agentic_core.agents.base_agent import MissionState


class TestWrongTestForDataType:
    """Scenario: The data type doesn't match the tests that were applied."""

    @pytest.mark.asyncio
    async def test_numerical_test_on_categorical_data(self):
        """Unpaired t-test applied to categorical SEX variable → ERROR."""
        state = MissionState(submission_id="wrong-1", job_id="j-1")
        state.agent_outputs = {
            "variables": [
                {"name": "SEX", "dtype": "categorical", "unique_values": 2, "missing_count": 0, "categories": ["M", "F"]},
                {"name": "BLOOD_GROUP", "dtype": "categorical", "unique_values": 4, "missing_count": 0, "categories": ["A", "B", "AB", "O"]},
            ],
            "sample_size": 80,
            "study_design": "two_group",
            "title": "Wrong Test Study",
            "test_results": [
                {"variable_name": "SEX", "test_result": {"test_type": "unpaired_t_test", "p_value": 0.12, "is_significant_05": False}},
                {"variable_name": "BLOOD_GROUP", "test_result": {"test_type": "one_way_anova", "p_value": 0.34, "is_significant_05": False}},
            ],
            "frequency_tables": [],
            "report_sections": [{"type": "paragraph", "content": "A study on 80 subjects."}],
            "charts": [],
        }

        agent = ReviewAgent(services=None)
        result = await agent.run(state)
        summary = result.agent_outputs["review_summary"]

        type_errors = [i for i in summary["items"] if i["check_id"] == "test_type_mismatch"]
        assert len(type_errors) == 2, f"Expected 2 type mismatches, got {len(type_errors)}"
        assert all(i["severity"] == "error" for i in type_errors)
        assert any("SEX" in i["variable"] for i in type_errors)
        assert any("BLOOD_GROUP" in i["variable"] for i in type_errors)
        print(f"\n✓ Detected {len(type_errors)} type mismatches:")
        for e in type_errors:
            print(f"  ❌ {e['message']}")
            print(f"     → {e['suggestion']}")

    @pytest.mark.asyncio
    async def test_categorical_test_on_numerical_data(self):
        """Chi-square and Z-test applied to continuous numerical variables → ERROR."""
        state = MissionState(submission_id="wrong-2", job_id="j-2")
        state.agent_outputs = {
            "variables": [
                {"name": "AGE", "dtype": "numerical", "unique_values": 40, "missing_count": 0, "categories": []},
                {"name": "WEIGHT", "dtype": "numerical", "unique_values": 35, "missing_count": 0, "categories": []},
            ],
            "sample_size": 100,
            "study_design": "single_cohort",
            "title": "Wrong Categorical Test",
            "test_results": [
                {"variable_name": "AGE", "test_result": {"test_type": "chi_square", "p_value": 0.04, "is_significant_05": True}},
                {"variable_name": "WEIGHT", "test_result": {"test_type": "z_test_proportion", "p_value": 0.01, "is_significant_05": True}},
            ],
            "frequency_tables": [],
            "report_sections": [{"type": "paragraph", "content": "A study on 100 subjects."}],
            "charts": [],
        }

        agent = ReviewAgent(services=None)
        result = await agent.run(state)
        summary = result.agent_outputs["review_summary"]

        type_errors = [i for i in summary["items"] if i["check_id"] == "test_type_mismatch"]
        assert len(type_errors) == 2
        assert all(i["severity"] == "error" for i in type_errors)
        assert any("AGE" in i["message"] and "chi_square" in i["message"] for i in type_errors)
        assert any("WEIGHT" in i["message"] and "z_test_proportion" in i["message"] for i in type_errors)
        print(f"\n✓ Detected categorical tests on numerical data:")
        for e in type_errors:
            print(f"  ❌ {e['message']}")

    @pytest.mark.asyncio
    async def test_parametric_on_verified_nonnormal(self):
        """t-test applied when Shapiro-Wilk explicitly shows non-normality → ERROR."""
        state = MissionState(submission_id="wrong-3", job_id="j-3")
        state.agent_outputs = {
            "variables": [
                {"name": "HOSPITAL_STAY", "dtype": "numerical", "unique_values": 20, "missing_count": 0, "categories": []},
            ],
            "sample_size": 60,
            "study_design": "two_group",
            "title": "Parametric on Non-Normal",
            "test_results": [
                {"variable_name": "HOSPITAL_STAY", "test_result": {"test_type": "shapiro_wilk", "p_value": 0.002, "is_significant_05": True}},
                {"variable_name": "HOSPITAL_STAY", "test_result": {"test_type": "unpaired_t_test", "p_value": 0.04, "is_significant_05": True}},
            ],
            "frequency_tables": [],
            "report_sections": [{"type": "paragraph", "content": "A study on 60 subjects."}],
            "charts": [],
        }

        agent = ReviewAgent(services=None)
        result = await agent.run(state)
        summary = result.agent_outputs["review_summary"]

        param_errors = [i for i in summary["items"] if i["check_id"] == "parametric_on_nonnormal"]
        assert len(param_errors) > 0
        assert param_errors[0]["severity"] == "error"
        assert "Mann-Whitney" in param_errors[0]["suggestion"]
        print(f"\n✓ Detected parametric on non-normal: {param_errors[0]['message']}")

    @pytest.mark.asyncio
    async def test_mixed_errors_verdict(self):
        """Multiple type mismatches should produce significant_issues verdict."""
        state = MissionState(submission_id="wrong-4", job_id="j-4")
        state.agent_outputs = {
            "variables": [
                {"name": "SEX", "dtype": "categorical", "unique_values": 2, "missing_count": 0, "categories": ["M", "F"]},
                {"name": "AGE", "dtype": "numerical", "unique_values": 40, "missing_count": 0, "categories": []},
                {"name": "BP", "dtype": "numerical", "unique_values": 30, "missing_count": 0, "categories": []},
            ],
            "sample_size": 50,
            "study_design": "two_group",
            "title": "Multiple Errors",
            "test_results": [
                {"variable_name": "SEX", "test_result": {"test_type": "unpaired_t_test", "p_value": 0.3, "is_significant_05": False}},
                {"variable_name": "AGE", "test_result": {"test_type": "chi_square", "p_value": 0.04, "is_significant_05": True}},
                {"variable_name": "BP", "test_result": {"test_type": "shapiro_wilk", "p_value": 0.001, "is_significant_05": True}},
                {"variable_name": "BP", "test_result": {"test_type": "one_way_anova", "p_value": 0.02, "is_significant_05": True}},
            ],
            "frequency_tables": [],
            "report_sections": [{"type": "paragraph", "content": "A study on 50 subjects."}],
            "charts": [],
        }

        agent = ReviewAgent(services=None)
        result = await agent.run(state)
        summary = result.agent_outputs["review_summary"]

        assert summary["errors"] >= 3
        assert summary["overall_verdict"] == "significant_issues"
        assert summary["estimated_review_minutes"] >= 30
        print(f"\n✓ Multiple errors → verdict: {summary['overall_verdict']}, errors: {summary['errors']}")


class TestInsignificantResultsRecommendations:
    """Scenario: Tests yield insignificant results and reviewer recommends alternatives."""

    @pytest.mark.asyncio
    async def test_borderline_p_suggests_trend(self):
        """p=0.06-0.09 should be flagged as borderline with suggestion."""
        state = MissionState(submission_id="insig-1", job_id="j-i1")
        state.agent_outputs = {
            "variables": [
                {"name": "HB_LEVEL", "dtype": "numerical", "unique_values": 30, "missing_count": 0, "categories": []},
                {"name": "PLATELET", "dtype": "numerical", "unique_values": 25, "missing_count": 0, "categories": []},
            ],
            "sample_size": 45,
            "study_design": "two_group",
            "title": "Borderline Results",
            "test_results": [
                {"variable_name": "HB_LEVEL", "test_result": {"test_type": "unpaired_t_test", "p_value": 0.062, "is_significant_05": False}},
                {"variable_name": "PLATELET", "test_result": {"test_type": "mann_whitney_u", "p_value": 0.078, "is_significant_05": False}},
            ],
            "frequency_tables": [],
            "report_sections": [{"type": "paragraph", "content": "A study on 45 subjects."}],
            "charts": [],
        }

        agent = ReviewAgent(services=None)
        result = await agent.run(state)
        summary = result.agent_outputs["review_summary"]

        borderline = [i for i in summary["items"] if i["check_id"] == "borderline_p"]
        assert len(borderline) == 2
        assert all("trend" in i["suggestion"].lower() for i in borderline)
        print(f"\n✓ Borderline p-values flagged:")
        for b in borderline:
            print(f"  ℹ️ {b['message']}")

    @pytest.mark.asyncio
    async def test_all_insignificant_multiple_testing(self):
        """Many insignificant tests → suggest reconsidering approach or sample size."""
        state = MissionState(submission_id="insig-2", job_id="j-i2")
        state.agent_outputs = {
            "variables": [
                {"name": f"VAR_{i}", "dtype": "numerical", "unique_values": 20, "missing_count": 0, "categories": []}
                for i in range(8)
            ],
            "sample_size": 30,
            "study_design": "two_group",
            "title": "All Insignificant",
            "test_results": [
                {"variable_name": f"VAR_{i}", "test_result": {"test_type": "unpaired_t_test", "p_value": 0.2 + i * 0.05, "is_significant_05": False}}
                for i in range(8)
            ],
            "frequency_tables": [],
            "report_sections": [{"type": "paragraph", "content": "A study on 30 subjects."}],
            "charts": [],
        }

        agent = ReviewAgent(services=None)
        result = await agent.run(state)
        summary = result.agent_outputs["review_summary"]

        # With 8 tests and none significant, multiple testing should still be flagged
        multiple = [i for i in summary["items"] if i["check_id"] == "multiple_testing"]
        assert len(multiple) > 0
        print(f"\n✓ Multiple testing flag: {multiple[0]['message']}")

    @pytest.mark.asyncio
    async def test_nonparametric_suggested_for_power(self):
        """Non-parametric on normal data flagged as losing power."""
        state = MissionState(submission_id="insig-3", job_id="j-i3")
        state.agent_outputs = {
            "variables": [
                {"name": "CHOLESTEROL", "dtype": "numerical", "unique_values": 40, "missing_count": 0, "categories": []},
            ],
            "sample_size": 100,
            "study_design": "two_group",
            "title": "Power Loss",
            "test_results": [
                {"variable_name": "CHOLESTEROL", "test_result": {"test_type": "shapiro_wilk", "p_value": 0.55, "is_significant_05": False}},
                {"variable_name": "CHOLESTEROL", "test_result": {"test_type": "mann_whitney_u", "p_value": 0.12, "is_significant_05": False}},
            ],
            "frequency_tables": [],
            "report_sections": [{"type": "paragraph", "content": "A study on 100 subjects."}],
            "charts": [],
        }

        agent = ReviewAgent(services=None)
        result = await agent.run(state)
        summary = result.agent_outputs["review_summary"]

        power_items = [i for i in summary["items"] if i["check_id"] == "nonparam_on_normal"]
        assert len(power_items) > 0
        assert "power" in power_items[0]["message"].lower()
        assert "t-test" in power_items[0]["suggestion"].lower()
        print(f"\n✓ Power improvement suggested: {power_items[0]['message']}")
        print(f"  → {power_items[0]['suggestion']}")


class TestInsufficientData:
    """Scenario: Masterchart has insufficient data for reliable statistical analysis."""

    @pytest.mark.asyncio
    async def test_very_small_sample_size(self):
        """N<10 should flag as insufficient for most tests."""
        np.random.seed(42)
        df = pd.DataFrame({
            "AGE": [23, 45, 34, 28, 51, 39, 42],
            "SEX": ["M", "M", "F", "M", "F", "M", "M"],
        })
        state = MissionState(submission_id="insuf-1", job_id="j-s1")
        state.agent_outputs = {
            "dataframe_json": df.to_json(),
            "variables": [
                {"name": "AGE", "dtype": "numerical", "unique_values": 7, "missing_count": 0, "categories": []},
                {"name": "SEX", "dtype": "categorical", "unique_values": 2, "missing_count": 0, "categories": ["M", "F"]},
            ],
            "sample_size": 7,
            "study_design": "single_cohort",
            "title": "Tiny Sample",
            "test_results": [
                {"variable_name": "AGE", "test_result": {"test_type": "unpaired_t_test", "p_value": 0.3, "is_significant_05": False}},
                {"variable_name": "SEX", "test_result": {"test_type": "chi_square", "p_value": 0.5, "is_significant_05": False}},
            ],
            "frequency_tables": [
                {"variable_name": "SEX", "rows": [
                    {"category": "M", "count": 5, "percentage": 71.4},
                    {"category": "F", "count": 2, "percentage": 28.6},
                ]},
            ],
            "report_sections": [{"type": "paragraph", "content": "A study on 7 subjects."}],
            "charts": [],
        }

        agent = ReviewAgent(services=None)
        result = await agent.run(state)
        summary = result.agent_outputs["review_summary"]

        sample_issues = [i for i in summary["items"] if "sample" in i["check_id"].lower() or "small_n" in i["check_id"]]
        assert len(sample_issues) > 0
        print(f"\n✓ Insufficient sample flagged:")
        for s in sample_issues:
            print(f"  ⚠️ {s['message']}")

    @pytest.mark.asyncio
    async def test_sparse_categories_insufficient(self):
        """Categories with <5 observations should flag chi-square as unreliable."""
        np.random.seed(42)
        df = pd.DataFrame({
            "BLOOD_TYPE": (["A"] * 3 + ["B"] * 2 + ["AB"] * 1 + ["O"] * 4),
            "OUTCOME": (["Good"] * 6 + ["Poor"] * 4),
        })
        state = MissionState(submission_id="insuf-2", job_id="j-s2")
        state.agent_outputs = {
            "dataframe_json": df.to_json(),
            "variables": [
                {"name": "BLOOD_TYPE", "dtype": "categorical", "unique_values": 4, "missing_count": 0, "categories": ["A", "B", "AB", "O"]},
                {"name": "OUTCOME", "dtype": "categorical", "unique_values": 2, "missing_count": 0, "categories": ["Good", "Poor"]},
            ],
            "sample_size": 10,
            "study_design": "two_group",
            "title": "Sparse Data",
            "test_results": [
                {"variable_name": "BLOOD_TYPE", "test_result": {"test_type": "chi_square", "p_value": 0.6, "is_significant_05": False}},
            ],
            "frequency_tables": [
                {"variable_name": "BLOOD_TYPE", "rows": [
                    {"category": "A", "count": 3, "percentage": 30.0},
                    {"category": "B", "count": 2, "percentage": 20.0},
                    {"category": "AB", "count": 1, "percentage": 10.0},
                    {"category": "O", "count": 4, "percentage": 40.0},
                ]},
            ],
            "report_sections": [{"type": "paragraph", "content": "A study on 10 subjects."}],
            "charts": [],
        }

        agent = ReviewAgent(services=None)
        result = await agent.run(state)
        summary = result.agent_outputs["review_summary"]

        chi_issues = [i for i in summary["items"] if "chi_sq" in i["check_id"]]
        assert len(chi_issues) > 0
        assert any("Fisher" in i["suggestion"] for i in chi_issues)
        print(f"\n✓ Sparse data flagged for chi-square:")
        for c in chi_issues:
            print(f"  ⚠️ {c['message']}")
            print(f"     → {c['suggestion']}")

    @pytest.mark.asyncio
    async def test_too_many_variables_for_sample(self):
        """Many variables with too few observations → overfitting risk."""
        state = MissionState(submission_id="insuf-3", job_id="j-s3")
        variables = [
            {"name": f"VAR_{i}", "dtype": "numerical", "unique_values": 10, "missing_count": 0, "categories": []}
            for i in range(15)
        ]
        test_results = [
            {"variable_name": f"VAR_{i}", "test_result": {"test_type": "unpaired_t_test", "p_value": 0.04 + i * 0.01, "is_significant_05": i < 3}}
            for i in range(15)
        ]
        state.agent_outputs = {
            "variables": variables,
            "sample_size": 20,
            "study_design": "two_group",
            "title": "Too Many Variables",
            "test_results": test_results,
            "frequency_tables": [],
            "report_sections": [{"type": "paragraph", "content": "A study on 20 subjects."}],
            "charts": [],
        }

        agent = ReviewAgent(services=None)
        result = await agent.run(state)
        summary = result.agent_outputs["review_summary"]

        # Should flag multiple testing AND sample adequacy
        issues = [i for i in summary["items"] if i["severity"] in ("warning", "error")]
        assert len(issues) > 0
        multiple = [i for i in summary["items"] if i["check_id"] == "multiple_testing"]
        assert len(multiple) > 0
        print(f"\n✓ Too many variables for sample flagged:")
        print(f"  Total issues: {len(issues)}")
        for m in multiple:
            print(f"  ℹ️ {m['message']}")

    @pytest.mark.asyncio
    async def test_high_missing_data_ratio(self):
        """Variables with lots of missing data → unreliable results."""
        np.random.seed(42)
        df = pd.DataFrame({
            "SCORE": [1, 2, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 3, 4,
                      np.nan, np.nan, np.nan, np.nan, np.nan, 5, np.nan, np.nan, np.nan, 6],
        })
        state = MissionState(submission_id="insuf-4", job_id="j-s4")
        state.agent_outputs = {
            "dataframe_json": df.to_json(),
            "variables": [
                {"name": "SCORE", "dtype": "numerical", "unique_values": 6, "missing_count": 14, "categories": []},
            ],
            "sample_size": 20,
            "study_design": "single_cohort",
            "title": "Missing Data",
            "test_results": [
                {"variable_name": "SCORE", "test_result": {"test_type": "shapiro_wilk", "p_value": 0.3, "is_significant_05": False}},
            ],
            "frequency_tables": [],
            "report_sections": [{"type": "paragraph", "content": "A study on 20 subjects."}],
            "charts": [],
        }

        agent = ReviewAgent(services=None)
        result = await agent.run(state)
        summary = result.agent_outputs["review_summary"]

        # With only 6 actual data points (14 missing out of 20), sample size mismatch should flag
        size_issues = [i for i in summary["items"] if "sample" in i["check_id"].lower()]
        assert len(size_issues) > 0
        print(f"\n✓ High missing data flagged:")
        for s in size_issues:
            print(f"  ⚠️ {s['message']}")
