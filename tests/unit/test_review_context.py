"""Unit tests for review context and data profile builder."""
import pytest
from unittest.mock import AsyncMock, MagicMock
from src.medstats.services.review_context import build_data_profile, get_latest_report


SAMPLE_STATE = {
    "study_design": "two_group",
    "sample_size": 84,
    "variables": [
        {"name": "AGE", "dtype": "numerical", "missing_pct": 0},
        {"name": "SEX", "dtype": "categorical", "n_categories": 2},
        {"name": "RECOVERY", "dtype": "numerical"},
    ],
    "test_results": [
        {"variable_name": "AGE", "test_result": {"test_type": "shapiro_wilk", "p_value": 0.21}},
        {
            "variable_name": "RECOVERY",
            "test_result": {"test_type": "unpaired_t_test", "p_value": 0.03},
        },
        {"variable_name": "SEX", "test_result": {"test_type": "chi_square", "p_value": 0.12}},
    ],
}


class TestBuildDataProfile:
    def test_basic_profile(self):
        profile = build_data_profile(SAMPLE_STATE)
        assert profile["study_design"] == "two_group"
        assert profile["sample_size"] == 84
        assert profile["variable_count"] == 3
        assert "Two-group" in profile["study_design_label"]
        assert "N=84" in profile["summary"]

    def test_variable_tests_and_normality(self):
        profile = build_data_profile(SAMPLE_STATE)
        by_name = {v["name"]: v for v in profile["variables"]}
        assert by_name["AGE"]["normality_ok"] is True
        assert by_name["RECOVERY"]["current_test"] == "unpaired_t_test"
        assert by_name["SEX"]["distribution_note"] == "binary"
        assert len(by_name["RECOVERY"]["valid_test_alternatives"]) > 0

    def test_empty_state(self):
        profile = build_data_profile({})
        assert profile["sample_size"] == 0
        assert profile["variables"] == []

    def test_duplicate_rows_prefers_inferential_test(self):
        state = {
            **SAMPLE_STATE,
            "variables": [
                *SAMPLE_STATE["variables"],
                {"name": "COMPLICATIONS", "dtype": "categorical", "n_categories": 2},
            ],
            "test_results": [
                *SAMPLE_STATE["test_results"],
                {
                    "variable_name": "SEX",
                    "descriptive": {"mean": None},
                    "test_result": {"test_type": "z_test_proportion", "p_value": 0.04},
                },
                {
                    "variable_name": "COMPLICATIONS",
                    "test_result": {"test_type": "fisher_exact", "p_value": 0.02},
                },
                {
                    "variable_name": "COMPLICATIONS",
                    "descriptive": {"mean": None},
                    "test_result": {"test_type": "z_test_proportion", "p_value": 0.04},
                },
            ],
        }
        profile = build_data_profile(state)
        by_name = {v["name"]: v for v in profile["variables"]}
        assert by_name["SEX"]["current_test"] == "chi_square"
        assert by_name["COMPLICATIONS"]["current_test"] == "fisher_exact"

    def test_single_cohort_categorical_includes_fisher_exact(self):
        state = {
            "study_design": "single_cohort",
            "sample_size": 40,
            "variables": [{"name": "COMPLICATIONS", "dtype": "categorical"}],
            "test_results": [
                {
                    "variable_name": "COMPLICATIONS",
                    "test_result": {"test_type": "z_test_proportion", "p_value": 0.1},
                },
            ],
        }
        profile = build_data_profile(state)
        alts = profile["variables"][0]["valid_test_alternatives"]
        values = {o["value"] for o in alts}
        assert "fisher_exact" in values
        assert "chi_square" in values
        assert "z_test_proportion" in values


class TestGetLatestReport:
    @pytest.mark.asyncio
    async def test_returns_first_scalar(self):
        report = MagicMock(version=2)
        result = MagicMock()
        result.scalars.return_value.first.return_value = report
        db = AsyncMock()
        db.execute = AsyncMock(return_value=result)

        latest = await get_latest_report(db, "sub-id")
        assert latest is report
        db.execute.assert_awaited_once()
