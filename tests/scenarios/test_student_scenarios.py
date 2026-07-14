"""
Student scenario tests — run locally without API/DB/Redis.

Generate fixtures first (once):
    python scripts/generate_scenario_fixtures.py

Run all scenarios:
    pytest tests/scenarios/ -v
    python scripts/run_scenario_tests.py
"""
import pytest

from tests.scenarios.definitions import STUDENT_SCENARIOS, StudentScenario
from tests.scenarios.pipeline_runner import run_scenario_pipeline
from tests.scenarios.report_validators import (
    assert_charts_exist,
    assert_chart_suffixes,
    assert_expected_tests,
    assert_report_structure,
    assert_summary_json,
    collect_test_types,
)


@pytest.fixture(params=STUDENT_SCENARIOS, ids=lambda s: s.id)
def scenario(request) -> StudentScenario:
    sc: StudentScenario = request.param
    assert sc.data_path().exists(), (
        f"Missing fixture {sc.data_path()}. Run: python scripts/generate_scenario_fixtures.py"
    )
    assert sc.intent_path().exists(), f"Missing intent file {sc.intent_path()}"
    return sc


class TestStudentScenarios:
    @pytest.mark.asyncio
    async def test_scenario_pipeline(self, scenario: StudentScenario, tmp_path, monkeypatch):
        import src.medstats.core.chart_generator as cg

        monkeypatch.setattr(cg, "CHART_DIR", str(tmp_path))

        state, meta = await run_scenario_pipeline(scenario, chart_dir=str(tmp_path))

        assert state.status == "COMPLETED", state.error_message
        assert meta["sample_size"] >= scenario.min_sample_size
        assert meta["study_design"] == scenario.study_design, (
            f"Expected design {scenario.study_design}, got {meta['study_design']}"
        )

        sections = meta["sections"]
        charts = meta["charts"]
        test_results = meta["test_results"]

        assert_report_structure(
            sections,
            min_variables=scenario.min_variables,
            require_charts=True,
        )
        assert_charts_exist(charts)

        if scenario.expected_tests:
            assert_expected_tests(test_results, scenario.expected_tests)

        if scenario.expected_chart_suffixes:
            paths = [c["path"] for c in charts]
            assert_chart_suffixes(paths, scenario.expected_chart_suffixes)

        summary = meta.get("summary") or ""
        if summary.strip().startswith("{"):
            assert_summary_json(summary)

        methodology = meta.get("methodology_note") or ""
        assert "Statistical analysis" in methodology or "statistical" in methodology.lower()

    @pytest.mark.asyncio
    async def test_all_statistical_test_types_covered(self, tmp_path, monkeypatch):
        """Across all scenarios, the pipeline should exercise every wired inferential test."""
        import src.medstats.core.chart_generator as cg

        monkeypatch.setattr(cg, "CHART_DIR", str(tmp_path))

        covered = set()
        for scenario in STUDENT_SCENARIOS:
            chart_subdir = tmp_path / scenario.id
            chart_subdir.mkdir(exist_ok=True)
            state, meta = await run_scenario_pipeline(scenario, chart_dir=str(chart_subdir))
            assert state.status == "COMPLETED", f"{scenario.id}: {state.error_message}"
            covered |= collect_test_types(meta["test_results"])

        expected_coverage = {
            "z_test_proportion",
            "shapiro_wilk",
            "unpaired_t_test",
            "mann_whitney_u",
            "chi_square",
            "fisher_exact",
            "one_way_anova",
            "kruskal_wallis",
        }
        missing = expected_coverage - covered
        assert not missing, f"Scenarios did not exercise: {missing}. Covered: {covered}"
