"""Unit tests for RevisionResolverAgent structured corrections."""
import pytest
from src.medstats.agents.revision_resolver.agent import RevisionResolverAgent
from src.medstats.agents.correction_agent.schemas import CorrectionType


VARIABLES = [
    {"name": "RECOVERY", "dtype": "numerical"},
    {"name": "SEX", "dtype": "categorical"},
]

AGENT_OUTPUTS = {
    "variables": VARIABLES,
    "test_results": [
        {"variable_name": "RECOVERY", "test_result": {"test_type": "unpaired_t_test"}},
        {"variable_name": "SEX", "test_result": {"test_type": "chi_square"}},
    ],
    "study_design": "two_group",
}


class TestRevisionResolverCorrections:
    def test_build_test_override_corrections(self):
        agent = RevisionResolverAgent(services=None)
        structured = {
            "test_overrides": [{"variable": "RECOVERY", "new_test": "mann_whitney_u"}],
            "section_feedback": [],
            "final_comment": "",
        }
        corrections = agent._build_corrections(structured, AGENT_OUTPUTS)
        assert len(corrections) == 1
        assert corrections[0]["type"] == CorrectionType.CHANGE_TEST
        assert corrections[0]["variable"] == "RECOVERY"
        assert corrections[0]["new_test"] == "mann_whitney_u"

    def test_build_section_feedback_rewrite(self):
        agent = RevisionResolverAgent(services=None)
        structured = {
            "test_overrides": [],
            "section_feedback": [
                {"target": "methodology", "preset_codes": ["unclear_wording", "cite_assumptions"]},
            ],
            "final_comment": "",
        }
        corrections = agent._build_corrections(structured, AGENT_OUTPUTS)
        assert len(corrections) == 1
        assert corrections[0]["type"] == CorrectionType.REWRITE_NARRATIVE

    def test_final_comment_fallback(self):
        agent = RevisionResolverAgent(services=None)
        structured = {
            "test_overrides": [],
            "section_feedback": [],
            "final_comment": "Please improve the discussion section.",
        }
        corrections = agent._build_corrections(structured, AGENT_OUTPUTS)
        assert len(corrections) == 1
        assert corrections[0]["type"] == CorrectionType.REWRITE_NARRATIVE

    def test_methodology_wrong_test_auto_suggests_nonparametric(self):
        agent = RevisionResolverAgent(services=None)
        structured = {
            "test_overrides": [],
            "section_feedback": [
                {"target": "methodology", "preset_codes": ["wrong_test", "study_design_mismatch"]},
            ],
            "final_comment": "",
        }
        corrections = agent._build_corrections(structured, AGENT_OUTPUTS)
        change_tests = [c for c in corrections if c["type"] == CorrectionType.CHANGE_TEST]
        assert any(c["variable"] == "RECOVERY" and c["new_test"] == "mann_whitney_u" for c in change_tests)

    def test_methodology_feedback_does_not_create_general_feedback(self):
        agent = RevisionResolverAgent(services=None)
        structured = {
            "test_overrides": [],
            "section_feedback": [
                {"target": "methodology", "preset_codes": ["sample_size_low", "study_design_mismatch"]},
            ],
            "final_comment": "",
        }
        corrections = agent._build_corrections(structured, AGENT_OUTPUTS)
        assert not any(c["type"] == CorrectionType.GENERAL_FEEDBACK for c in corrections)

    def test_refresh_methodology_uses_test_results(self):
        agent = RevisionResolverAgent(services=None)
        state = type("State", (), {"agent_outputs": {
            "test_results": [
                {"variable_name": "RECOVERY", "test_result": {"test_type": "mann_whitney_u"}},
            ],
            "sample_size": 50,
            "report_sections": [
                {"type": "heading", "content": "Statistical Analysis"},
                {"type": "paragraph", "content": "Old methodology with Z-test of proportions"},
            ],
        }})()
        agent._refresh_methodology_section(state)
        methodology = state.agent_outputs["report_sections"][1]["content"]
        assert "Mann-Whitney" in methodology or "mann" in methodology.lower()
        assert "Professor" not in methodology

    @pytest.mark.asyncio
    async def test_run_with_test_override(self):
        import pandas as pd
        import numpy as np
        from src.agentic_core.agents.state import MissionState

        np.random.seed(42)
        df = pd.DataFrame({
            "GROUP": ["A"] * 25 + ["B"] * 25,
            "RECOVERY": np.concatenate([
                np.random.normal(10, 2, 25),
                np.random.normal(12, 2, 25),
            ]),
        })

        state = MissionState(submission_id="rev-test", job_id="job-rev")
        state.agent_outputs = {
            "dataframe_json": df.to_json(),
            "variables": [
                {"name": "RECOVERY", "dtype": "numerical"},
            ],
            "groups": {"column": "GROUP", "groups": ["A", "B"]},
            "study_design": "two_group",
            "sample_size": 50,
            "title": "Recovery Study",
            "intent": "Compare recovery between groups",
            "test_results": [
                {
                    "variable_name": "RECOVERY",
                    "test_result": {
                        "test_type": "unpaired_t_test",
                        "p_value": 0.04,
                        "is_significant_05": True,
                        "significance_statement": "Significant difference (p=0.04)",
                    },
                },
            ],
            "frequency_tables": [],
            "report_sections": [
                {"type": "heading", "content": "OBSERVATIONS AND RESULTS"},
                {"type": "table_heading", "content": "Table No.1: RECOVERY", "variable_name": "RECOVERY"},
                {"type": "test_result", "content": "Significant difference (p=0.04)"},
            ],
            "report_version": 1,
            "structured_review": {
                "test_overrides": [{"variable": "RECOVERY", "new_test": "mann_whitney_u"}],
                "section_feedback": [],
                "final_comment": "",
            },
        }

        agent = RevisionResolverAgent(services=None)
        result = await agent.run(state)
        assert result.status != "FAILED"
        assert result.agent_outputs["report_version"] == 2
        tr = result.agent_outputs["test_results"][0]["test_result"]
        assert tr["test_type"] == "mann_whitney_u"
