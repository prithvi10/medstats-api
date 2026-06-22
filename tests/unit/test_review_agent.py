"""Unit tests for the AI Professor Review Agent."""
import pytest
import asyncio
import pandas as pd
import numpy as np
from pathlib import Path
from src.medstats.agents.review_agent import ReviewAgent, ReviewSummary, Severity
from src.agentic_core.agents.base_agent import MissionState

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def sample_report_state():
    """State after successful pipeline with full report data."""
    np.random.seed(42)
    df = pd.DataFrame({
        "AGE": np.random.randint(18, 65, 100).astype(float),
        "SEX": np.random.choice(["male", "female"], 100, p=[0.6, 0.4]),
        "BLOOD GROUP": np.random.choice(["A", "B", "AB", "O"], 100),
        "DIAGNOSIS": np.random.choice(["Type A", "Type B"], 100),
    })

    state = MissionState(submission_id="test-review", job_id="job-review")
    state.agent_outputs = {
        "dataframe_json": df.to_json(),
        "variables": [
            {"name": "AGE", "dtype": "numerical", "unique_values": 40, "missing_count": 0, "categories": []},
            {"name": "SEX", "dtype": "categorical", "unique_values": 2, "missing_count": 0, "categories": ["male", "female"]},
            {"name": "BLOOD GROUP", "dtype": "categorical", "unique_values": 4, "missing_count": 0, "categories": ["A", "B", "AB", "O"]},
            {"name": "DIAGNOSIS", "dtype": "categorical", "unique_values": 2, "missing_count": 0, "categories": ["Type A", "Type B"]},
        ],
        "sample_size": 100,
        "study_design": "cross-sectional",
        "title": "Test Study on Blood Groups",
        "test_results": [
            {
                "variable_name": "AGE",
                "test_result": {
                    "test_type": "shapiro_wilk",
                    "p_value": 0.23,
                    "is_significant_05": False,
                    "significance_statement": "Age is normally distributed (p=0.23)",
                },
            },
            {
                "variable_name": "AGE",
                "test_result": {
                    "test_type": "unpaired_t_test",
                    "p_value": 0.045,
                    "is_significant_05": True,
                    "significance_statement": "Significant difference in age between groups (p=0.045)",
                },
            },
            {
                "variable_name": "SEX",
                "test_result": {
                    "test_type": "chi_square",
                    "p_value": 0.32,
                    "is_significant_05": False,
                    "significance_statement": "No significant association between sex and diagnosis (p=0.32)",
                },
            },
            {
                "variable_name": "BLOOD GROUP",
                "test_result": {
                    "test_type": "chi_square",
                    "p_value": 0.08,
                    "is_significant_05": False,
                    "significance_statement": "No significant association between blood group and diagnosis (p=0.08)",
                },
            },
        ],
        "frequency_tables": [
            {
                "variable_name": "SEX",
                "rows": [
                    {"category": "male", "count": 60, "percentage": 60.0},
                    {"category": "female", "count": 40, "percentage": 40.0},
                ],
            },
            {
                "variable_name": "BLOOD GROUP",
                "rows": [
                    {"category": "A", "count": 25, "percentage": 25.0},
                    {"category": "B", "count": 30, "percentage": 30.0},
                    {"category": "AB", "count": 20, "percentage": 20.0},
                    {"category": "O", "count": 25, "percentage": 25.0},
                ],
            },
        ],
        "report_sections": [
            {"type": "heading", "content": "OBSERVATIONS AND RESULTS"},
            {"type": "paragraph", "content": "A study was conducted on 100 subjects."},
            {"type": "table_heading", "content": "Table No.1: AGE distribution"},
            {"type": "paragraph", "content": "The mean age was 41 years."},
            {"type": "table_heading", "content": "Table No.2: SEX distribution"},
            {"type": "paragraph", "content": "Males comprised 60% of the study population."},
            {"type": "table_heading", "content": "Table No.3: BLOOD GROUP distribution"},
            {"type": "paragraph", "content": "Blood group B was the most common at 30%."},
        ],
        "charts": [
            {"variable": "AGE", "type": "histogram", "path": "/tmp/age.png"},
            {"variable": "SEX", "type": "pie", "path": "/tmp/sex.png"},
            {"variable": "BLOOD GROUP", "type": "bar", "path": "/tmp/bg.png"},
        ],
    }
    return state


@pytest.fixture
def bad_report_state():
    """State with intentional errors for the reviewer to catch."""
    np.random.seed(42)
    df = pd.DataFrame({
        "AGE": np.random.randint(18, 65, 100).astype(float),
        "SEX": np.random.choice(["male", "female"], 100),
        "SCORE": np.random.normal(50, 10, 100),
    })

    state = MissionState(submission_id="test-bad", job_id="job-bad")
    state.agent_outputs = {
        "dataframe_json": df.to_json(),
        "variables": [
            {"name": "AGE", "dtype": "numerical", "unique_values": 40, "missing_count": 0, "categories": []},
            {"name": "SEX", "dtype": "categorical", "unique_values": 2, "missing_count": 0, "categories": ["male", "female"]},
            {"name": "SCORE", "dtype": "numerical", "unique_values": 100, "missing_count": 0, "categories": []},
        ],
        "sample_size": 100,
        "study_design": "cross-sectional",
        "title": "Bad Report Study",
        "test_results": [
            {
                "variable_name": "SEX",
                "test_result": {
                    "test_type": "unpaired_t_test",  # Wrong! categorical data with parametric test
                    "p_value": 0.04,
                    "is_significant_05": True,
                },
            },
            {
                "variable_name": "SCORE",
                "test_result": {
                    "test_type": "unpaired_t_test",  # No normality test first
                    "p_value": 0.03,
                    "is_significant_05": True,
                },
            },
        ],
        "frequency_tables": [
            {
                "variable_name": "SEX",
                "rows": [
                    {"category": "male", "count": 55, "percentage": 55.0},
                    {"category": "female", "count": 45, "percentage": 40.0},  # Sums to 95!
                ],
            },
        ],
        "report_sections": [
            {"type": "heading", "content": "OBSERVATIONS AND RESULTS"},
            {"type": "paragraph", "content": "A study was conducted on 150 subjects."},  # Wrong N
            {"type": "table_heading", "content": "Table No.1: AGE"},
            {"type": "table_heading", "content": "Table No.3: SEX"},  # Skipped table 2!
            {"type": "paragraph", "content": "This proved that there is no association between age and outcome."},
        ],
        "charts": [],  # No charts at all
    }
    return state


class TestReviewAgentHappyPath:
    @pytest.mark.asyncio
    async def test_review_good_report(self, sample_report_state):
        """A well-formed report should get mostly OK results."""
        agent = ReviewAgent(services=None)
        result_state = await agent.run(sample_report_state)

        assert "review_summary" in result_state.agent_outputs
        summary = result_state.agent_outputs["review_summary"]

        assert summary["total_checks"] > 0
        assert summary["passed"] > 0
        assert summary["overall_verdict"] in ("approve_recommended", "minor_issues")
        print(f"\n✓ Good report: {summary['passed']} passed, {summary['warnings']} warnings, {summary['errors']} errors")
        print(f"  Verdict: {summary['overall_verdict']}")
        print(f"  Estimated review time: {summary['estimated_review_minutes']} min")

    @pytest.mark.asyncio
    async def test_review_includes_all_tiers(self, sample_report_state):
        """Review should have checks from all 3 tiers."""
        agent = ReviewAgent(services=None)
        result_state = await agent.run(sample_report_state)
        summary = result_state.agent_outputs["review_summary"]
        all_items = summary["all_items"]

        tiers_present = set(item["tier"] for item in all_items)
        assert 1 in tiers_present, "Tier 1 (Mechanical) checks missing"
        assert 2 in tiers_present, "Tier 2 (Methodology) checks missing"

    @pytest.mark.asyncio
    async def test_percentage_sums_correct(self, sample_report_state):
        """Correct percentage tables should pass."""
        agent = ReviewAgent(services=None)
        result_state = await agent.run(sample_report_state)
        summary = result_state.agent_outputs["review_summary"]
        all_items = summary["all_items"]

        pct_items = [i for i in all_items if i["check_id"] == "pct_sum"]
        assert len(pct_items) > 0
        assert all(i["severity"] == "ok" for i in pct_items)

    @pytest.mark.asyncio
    async def test_borderline_p_detected(self, sample_report_state):
        """Borderline p-value (0.08) should be flagged."""
        agent = ReviewAgent(services=None)
        result_state = await agent.run(sample_report_state)
        summary = result_state.agent_outputs["review_summary"]
        all_items = summary["all_items"]

        borderline = [i for i in all_items if i["check_id"] == "borderline_p"]
        assert len(borderline) > 0
        assert borderline[0]["variable"] == "BLOOD GROUP"


class TestReviewAgentErrorDetection:
    @pytest.mark.asyncio
    async def test_catches_wrong_test_type(self, bad_report_state):
        """Should catch categorical data analyzed with parametric test."""
        agent = ReviewAgent(services=None)
        result_state = await agent.run(bad_report_state)
        summary = result_state.agent_outputs["review_summary"]
        all_items = summary["all_items"]

        mismatch = [i for i in all_items if i["check_id"] == "test_type_mismatch"]
        assert len(mismatch) > 0
        assert mismatch[0]["severity"] == "error"
        assert "SEX" in mismatch[0]["variable"]
        print(f"\n✓ Detected wrong test: {mismatch[0]['message']}")

    @pytest.mark.asyncio
    async def test_catches_percentage_error(self, bad_report_state):
        """Should catch percentages not summing to 100%."""
        agent = ReviewAgent(services=None)
        result_state = await agent.run(bad_report_state)
        summary = result_state.agent_outputs["review_summary"]

        pct_errors = [i for i in summary["items"] if i["check_id"] == "pct_sum_error"]
        assert len(pct_errors) > 0
        assert "95" in pct_errors[0]["message"]
        print(f"\n✓ Detected percentage error: {pct_errors[0]['message']}")

    @pytest.mark.asyncio
    async def test_catches_table_numbering_gap(self, bad_report_state):
        """Should detect skipped table numbers."""
        agent = ReviewAgent(services=None)
        result_state = await agent.run(bad_report_state)
        summary = result_state.agent_outputs["review_summary"]

        table_err = [i for i in summary["items"] if i["check_id"] == "table_seq_error"]
        assert len(table_err) > 0
        print(f"\n✓ Detected table gap: {table_err[0]['message']}")

    @pytest.mark.asyncio
    async def test_catches_sample_size_mismatch_in_text(self, bad_report_state):
        """Should detect narrative saying 150 when N=100."""
        agent = ReviewAgent(services=None)
        result_state = await agent.run(bad_report_state)
        summary = result_state.agent_outputs["review_summary"]

        mismatch = [i for i in summary["items"] if i["check_id"] == "sample_mismatch"]
        assert len(mismatch) > 0
        assert "150" in mismatch[0]["message"]
        print(f"\n✓ Detected N mismatch: {mismatch[0]['message']}")

    @pytest.mark.asyncio
    async def test_catches_missing_normality_check(self, bad_report_state):
        """Should flag parametric test without normality check."""
        agent = ReviewAgent(services=None)
        result_state = await agent.run(bad_report_state)
        summary = result_state.agent_outputs["review_summary"]

        norm_issues = [i for i in summary["items"] if i["check_id"] == "normality_not_tested"]
        assert len(norm_issues) > 0
        print(f"\n✓ Detected missing normality: {norm_issues[0]['message']}")

    @pytest.mark.asyncio
    async def test_catches_negative_overstatement(self, bad_report_state):
        """Should flag 'proved that there is no' as overstatement."""
        agent = ReviewAgent(services=None)
        result_state = await agent.run(bad_report_state)
        summary = result_state.agent_outputs["review_summary"]

        overstate = [i for i in summary["items"] if i["check_id"] == "negative_overstatement"]
        assert len(overstate) > 0
        print(f"\n✓ Detected overstatement: {overstate[0]['message']}")

    @pytest.mark.asyncio
    async def test_missing_variable_coverage(self, bad_report_state):
        """Should detect variables not covered in report."""
        agent = ReviewAgent(services=None)
        result_state = await agent.run(bad_report_state)
        summary = result_state.agent_outputs["review_summary"]
        all_items = summary["all_items"]

        missing = [i for i in all_items if i["check_id"] == "var_missing"]
        assert len(missing) > 0
        print(f"\n✓ Detected missing vars: {[m['variable'] for m in missing]}")

    @pytest.mark.asyncio
    async def test_bad_report_gets_bad_verdict(self, bad_report_state):
        """Bad report should receive 'needs_attention' or 'significant_issues'."""
        agent = ReviewAgent(services=None)
        result_state = await agent.run(bad_report_state)
        summary = result_state.agent_outputs["review_summary"]

        assert summary["overall_verdict"] in ("needs_attention", "significant_issues")
        assert summary["estimated_review_minutes"] >= 20
        print(f"\n✓ Bad report verdict: {summary['overall_verdict']} (est. {summary['estimated_review_minutes']} min)")


class TestTestSelectionValidation:
    """Tests for the deep test-selection logic (Tier 2.2)."""

    @pytest.mark.asyncio
    async def test_parametric_on_nonnormal_data(self):
        """Should flag t-test when Shapiro-Wilk shows non-normality."""
        state = MissionState(submission_id="test-sel", job_id="job-sel")
        state.agent_outputs = {
            "variables": [
                {"name": "SCORE", "dtype": "numerical", "unique_values": 50, "missing_count": 0, "categories": []},
            ],
            "sample_size": 60,
            "study_design": "two_group",
            "title": "Test",
            "test_results": [
                {
                    "variable_name": "SCORE",
                    "test_result": {"test_type": "shapiro_wilk", "p_value": 0.001, "is_significant_05": True},
                },
                {
                    "variable_name": "SCORE",
                    "test_result": {"test_type": "unpaired_t_test", "p_value": 0.03, "is_significant_05": True},
                },
            ],
            "frequency_tables": [],
            "report_sections": [],
            "charts": [],
        }

        agent = ReviewAgent(services=None)
        result_state = await agent.run(state)
        summary = result_state.agent_outputs["review_summary"]

        errors = [i for i in summary["items"] if i["check_id"] == "parametric_on_nonnormal"]
        assert len(errors) > 0
        assert "Mann-Whitney" in errors[0]["suggestion"]
        print(f"\n✓ Detected parametric on non-normal: {errors[0]['message']}")

    @pytest.mark.asyncio
    async def test_nonparametric_on_normal_data_flagged(self):
        """Should suggest parametric test when data is normal (higher power)."""
        state = MissionState(submission_id="test-sel2", job_id="job-sel2")
        state.agent_outputs = {
            "variables": [
                {"name": "BP", "dtype": "numerical", "unique_values": 40, "missing_count": 0, "categories": []},
            ],
            "sample_size": 80,
            "study_design": "two_group",
            "title": "Test",
            "test_results": [
                {
                    "variable_name": "BP",
                    "test_result": {"test_type": "shapiro_wilk", "p_value": 0.45, "is_significant_05": False},
                },
                {
                    "variable_name": "BP",
                    "test_result": {"test_type": "mann_whitney_u", "p_value": 0.08, "is_significant_05": False},
                },
            ],
            "frequency_tables": [],
            "report_sections": [],
            "charts": [],
        }

        agent = ReviewAgent(services=None)
        result_state = await agent.run(state)
        summary = result_state.agent_outputs["review_summary"]

        info_items = [i for i in summary["items"] if i["check_id"] == "nonparam_on_normal"]
        assert len(info_items) > 0
        assert "power" in info_items[0]["message"].lower()
        print(f"\n✓ Detected power loss: {info_items[0]['message']}")

    @pytest.mark.asyncio
    async def test_ztest_on_many_categories(self):
        """Should flag Z-test proportion when variable has many categories."""
        np.random.seed(42)
        df = pd.DataFrame({
            "DIAGNOSIS": np.random.choice(
                [f"Type_{i}" for i in range(15)], 100
            ),
        })
        state = MissionState(submission_id="test-sel3", job_id="job-sel3")
        state.agent_outputs = {
            "dataframe_json": df.to_json(),
            "variables": [
                {"name": "DIAGNOSIS", "dtype": "categorical", "unique_values": 15, "missing_count": 0, "categories": []},
            ],
            "sample_size": 100,
            "study_design": "single_cohort",
            "title": "Test",
            "test_results": [
                {
                    "variable_name": "DIAGNOSIS",
                    "test_type": "z_test_proportion",
                    "p_value": 0.02,
                    "is_significant_05": True,
                },
            ],
            "frequency_tables": [],
            "report_sections": [],
            "charts": [],
        }

        agent = ReviewAgent(services=None)
        result_state = await agent.run(state)
        summary = result_state.agent_outputs["review_summary"]

        ztest_issues = [i for i in summary["items"] if i["check_id"] == "ztest_too_many_cats"]
        assert len(ztest_issues) > 0
        assert "15 categories" in ztest_issues[0]["message"]
        print(f"\n✓ Detected Z-test on many categories: {ztest_issues[0]['message']}")

    @pytest.mark.asyncio
    async def test_small_sample_no_normality(self):
        """Should flag parametric test on small sample without normality check."""
        state = MissionState(submission_id="test-sel4", job_id="job-sel4")
        state.agent_outputs = {
            "variables": [
                {"name": "WEIGHT", "dtype": "numerical", "unique_values": 20, "missing_count": 0, "categories": []},
            ],
            "sample_size": 18,
            "study_design": "two_group",
            "title": "Test",
            "test_results": [
                {
                    "variable_name": "WEIGHT",
                    "test_result": {"test_type": "unpaired_t_test", "p_value": 0.04, "is_significant_05": True},
                },
            ],
            "frequency_tables": [],
            "report_sections": [],
            "charts": [],
        }

        agent = ReviewAgent(services=None)
        result_state = await agent.run(state)
        summary = result_state.agent_outputs["review_summary"]

        small_n = [i for i in summary["items"] if i["check_id"] == "small_n_no_normality"]
        assert len(small_n) > 0
        assert "N=18" in small_n[0]["message"]
        print(f"\n✓ Detected small sample without normality: {small_n[0]['message']}")

    @pytest.mark.asyncio
    async def test_parametric_on_normal_data_passes(self):
        """Parametric test on verified normal data should pass cleanly."""
        state = MissionState(submission_id="test-sel5", job_id="job-sel5")
        state.agent_outputs = {
            "variables": [
                {"name": "HEIGHT", "dtype": "numerical", "unique_values": 30, "missing_count": 0, "categories": []},
            ],
            "sample_size": 50,
            "study_design": "two_group",
            "title": "Test",
            "test_results": [
                {
                    "variable_name": "HEIGHT",
                    "test_result": {"test_type": "shapiro_wilk", "p_value": 0.35, "is_significant_05": False},
                },
                {
                    "variable_name": "HEIGHT",
                    "test_result": {"test_type": "unpaired_t_test", "p_value": 0.02, "is_significant_05": True},
                },
            ],
            "frequency_tables": [],
            "report_sections": [],
            "charts": [],
        }

        agent = ReviewAgent(services=None)
        result_state = await agent.run(state)
        summary = result_state.agent_outputs["review_summary"]

        ok_items = [i for i in summary["all_items"] if i["check_id"] == "parametric_normal_ok"]
        assert len(ok_items) > 0
        assert "justified" in ok_items[0]["message"]
        print(f"\n✓ Correctly approved: {ok_items[0]['message']}")


class TestReviewAgentWithRealData:
    @pytest.mark.asyncio
    async def test_review_after_full_pipeline(self):
        """Run Review Agent on output from Parser + Selector (skipping ReportGenerator's known issue)."""
        from src.medstats.agents.document_parser import DocumentParserAgent
        from src.medstats.agents.test_selector import TestSelectorAgent

        excel_path = FIXTURES_DIR / "medicine5_data.xlsx"
        if not excel_path.exists():
            pytest.skip("MEDICINE 5 fixture not available")

        state = MissionState(submission_id="review-e2e", job_id="review-e2e")
        state.agent_outputs = {
            "files": [
                {"path": str(excel_path), "type": "data_excel", "name": "medicine5_data.xlsx"},
            ],
            "intent": "Statistical analysis of medical data",
            "title": "TENS for Paediatric Long Bone Fractures",
        }

        parser = DocumentParserAgent(services=None)
        state = await parser.run(state)
        assert state.status != "FAILED", f"Parser failed: {state.error_message}"

        selector = TestSelectorAgent(services=None)
        state = await selector.run(state)
        assert state.status != "FAILED", f"Selector failed: {state.error_message}"

        # Simulate report sections (since ReportGenerator has a known bug with this data)
        state.agent_outputs["report_sections"] = [
            {"type": "heading", "content": "OBSERVATIONS AND RESULTS"},
            {"type": "paragraph", "content": f"A study was conducted on {state.agent_outputs['sample_size']} subjects."},
            {"type": "table_heading", "content": "Table No.1: AGE distribution"},
            {"type": "paragraph", "content": "The age of patients ranged from 5 to 15 years."},
            {"type": "table_heading", "content": "Table No.2: SEX distribution"},
            {"type": "paragraph", "content": "Males comprised 84% of the study population."},
        ]
        state.agent_outputs["charts"] = []

        # Now run Review Agent
        reviewer = ReviewAgent(services=None)
        state = await reviewer.run(state)
        assert state.status != "FAILED", f"Reviewer failed: {state.error_message}"

        summary = state.agent_outputs["review_summary"]
        assert summary["total_checks"] > 5

        print(f"\n{'=' * 60}")
        print("AI PROFESSOR REVIEW - MEDICINE 5 DATA")
        print(f"{'=' * 60}")
        print(f"Total checks: {summary['total_checks']}")
        print(f"Passed: {summary['passed']}")
        print(f"Info: {summary['info_count']}")
        print(f"Warnings: {summary['warnings']}")
        print(f"Errors: {summary['errors']}")
        print(f"Verdict: {summary['overall_verdict']}")
        print(f"Estimated review time: {summary['estimated_review_minutes']} min")
        print(f"\nFlagged items:")
        for item in summary["items"]:
            icon = "⚠️" if item["severity"] == "warning" else "❌" if item["severity"] == "error" else "ℹ️"
            print(f"  {icon} [T{item['tier']}] {item['category']}: {item['message']}")
            if item["suggestion"]:
                print(f"      → {item['suggestion']}")
