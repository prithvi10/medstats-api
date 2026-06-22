"""Unit tests for the Correction Agent."""
import pytest
import asyncio
import pandas as pd
import numpy as np
from pathlib import Path
from src.medstats.agents.correction_agent import CorrectionAgent, CorrectionType
from src.agentic_core.agents.base_agent import MissionState

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def sample_state_with_results():
    """State after a successful pipeline run."""
    np.random.seed(42)
    df = pd.DataFrame({
        "AGE": np.random.randint(5, 16, 50).astype(float),
        "SEX": np.random.choice(["male", "female"], 50, p=[0.84, 0.16]),
        "MECHANISM OF INJURY": np.random.choice(
            ["RTA", "Simple fall", "Fall from height", "Sports"], 50
        ),
    })

    state = MissionState(submission_id="test-corr", job_id="job-corr")
    state.agent_outputs = {
        "parsed_dataset": df.to_json(),
        "variables": [
            {"name": "AGE", "dtype": "numerical", "unique_values": 11, "missing_count": 0, "categories": []},
            {"name": "SEX", "dtype": "categorical", "unique_values": 2, "missing_count": 0, "categories": ["male", "female"]},
            {"name": "MECHANISM OF INJURY", "dtype": "categorical", "unique_values": 4, "missing_count": 0, "categories": []},
        ],
        "sample_size": 50,
        "title": "Test Study",
        "test_results": [
            {"variable_name": "AGE", "test_result": {"test_type": "shapiro_wilk", "p_value": 0.12, "is_significant_05": False}},
            {"variable_name": "SEX", "test_result": {"test_type": "z_test_proportion", "p_value": 0.001, "is_significant_05": True, "is_significant_01": True}},
        ],
        "frequency_tables": [
            {"variable_name": "SEX", "rows": [{"category": "male", "count": 42, "percentage": 84.0}, {"category": "female", "count": 8, "percentage": 16.0}]},
        ],
        "report_sections": [
            {"type": "heading", "content": "OBSERVATIONS AND RESULTS"},
            {"type": "paragraph", "content": "A study was conducted on 50 subjects."},
            {"type": "table_heading", "content": "Table No.1: AGE"},
            {"type": "paragraph", "content": "The age ranged from 5 to 15 years."},
            {"type": "test_result", "content": "Age is normally distributed (p=0.12)."},
        ],
        "report_version": 1,
    }
    return state


class TestCorrectionClassification:
    @pytest.mark.asyncio
    async def test_classify_test_change(self, sample_state_with_results):
        agent = CorrectionAgent(services=None)
        sample_state_with_results.agent_outputs["professor_feedback"] = "Use chi-square test for SEX variable"
        sample_state_with_results.agent_outputs["feedback_type"] = "general_feedback"

        result = await agent.run(sample_state_with_results)
        assert result.status != "FAILED"
        assert result.agent_outputs["report_version"] == 2
        history = result.agent_outputs["correction_history"]
        assert len(history) == 1
        assert any("chi_square" in str(c) for c in history[0]["corrections_applied"])

    @pytest.mark.asyncio
    async def test_classify_exclusion(self, sample_state_with_results):
        agent = CorrectionAgent(services=None)
        sample_state_with_results.agent_outputs["professor_feedback"] = "Exclude patients under 7 years of age"
        sample_state_with_results.agent_outputs["feedback_type"] = "general_feedback"

        result = await agent.run(sample_state_with_results)
        assert result.status != "FAILED"
        assert result.agent_outputs["report_version"] == 2

        history = result.agent_outputs["correction_history"]
        applied = history[0]["corrections_applied"]
        exclusion = next((a for a in applied if a.get("type") == "exclude_data"), None)
        assert exclusion is not None
        assert exclusion["new_size"] < exclusion["original_size"]

    @pytest.mark.asyncio
    async def test_classify_rewrite(self, sample_state_with_results):
        agent = CorrectionAgent(services=None)
        sample_state_with_results.agent_outputs["professor_feedback"] = "Rewrite the AGE paragraph"
        sample_state_with_results.agent_outputs["feedback_type"] = "general_feedback"

        result = await agent.run(sample_state_with_results)
        assert result.status != "FAILED"
        assert result.agent_outputs["report_version"] == 2

    @pytest.mark.asyncio
    async def test_explicit_feedback_type(self, sample_state_with_results):
        agent = CorrectionAgent(services=None)
        sample_state_with_results.agent_outputs["professor_feedback"] = "Run z-test on SEX"
        sample_state_with_results.agent_outputs["feedback_type"] = CorrectionType.RERUN_TEST
        sample_state_with_results.agent_outputs["target_variables"] = ["SEX"]

        result = await agent.run(sample_state_with_results)
        assert result.status != "FAILED"
        assert result.agent_outputs["report_version"] == 2


class TestCorrectionWithRealData:
    @pytest.mark.asyncio
    async def test_correction_after_full_pipeline(self):
        """Run pipeline then apply correction on MEDICINE 5 data."""
        from src.medstats.agents.document_parser import DocumentParserAgent
        from src.medstats.agents.test_selector import TestSelectorAgent
        from src.medstats.agents.report_generator import ReportGeneratorAgent
        from src.agentic_core.agents.orchestrator import MissionControl

        excel_path = str(FIXTURES_DIR / "medicine5_data.xlsx")
        discussion_path = str(FIXTURES_DIR / "DISCUSSION.docx")

        # Run initial pipeline
        state = MissionState(submission_id="corr-e2e", job_id="corr-job")
        state.agent_outputs["files"] = [
            {"path": excel_path, "type": "data_excel", "name": "medicine5_data.xlsx"},
            {"path": discussion_path, "type": "supporting_doc", "name": "DISCUSSION.docx"},
        ]
        state.agent_outputs["intent"] = "Evaluate TENS outcomes"
        state.agent_outputs["title"] = "TENS Study"

        agents = [DocumentParserAgent(services=None), TestSelectorAgent(services=None), ReportGeneratorAgent(services=None)]
        orchestrator = MissionControl(agents=agents)
        async for s in orchestrator.execute(state):
            state = s
        assert state.status == "COMPLETED"

        original_sample = state.agent_outputs["sample_size"]
        original_version = state.agent_outputs.get("report_version", 1)

        # Apply correction: exclude young patients
        state.agent_outputs["professor_feedback"] = "Exclude patients under 7 years old"
        state.agent_outputs["feedback_type"] = "general_feedback"

        correction_agent = CorrectionAgent(services=None)
        corrected_state = await correction_agent.run(state)

        assert corrected_state.status != "FAILED"
        assert corrected_state.agent_outputs["report_version"] == original_version + 1
        assert corrected_state.agent_outputs["sample_size"] < original_sample

        print(f"\n  Original sample: {original_sample}")
        print(f"  After correction: {corrected_state.agent_outputs['sample_size']}")
        print(f"  Version: {corrected_state.agent_outputs['report_version']}")
