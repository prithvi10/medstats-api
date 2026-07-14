"""
Integration tests for professor HITL review → RevisionResolver → updated report.

Verifies that structured professor feedback actually changes tests, methodology,
executive summary, and report sections — without injecting raw professor notes.
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
import pytest

from src.agentic_core.agents.state import MissionState
from src.medstats.agents.revision_resolver import RevisionResolverAgent
from src.medstats.agents.report_generator import ReportGeneratorAgent


def _two_group_state() -> MissionState:
    np.random.seed(7)
    df = pd.DataFrame({
        "GROUP": ["Control"] * 30 + ["Treatment"] * 30,
        "RECOVERY DAYS": np.concatenate([
            np.random.normal(14, 4, 30),
            np.random.normal(18, 5, 30),
        ]),
        "RECOVERED": ["Yes"] * 20 + ["No"] * 10 + ["Yes"] * 12 + ["No"] * 18,
    })

    state = MissionState(submission_id="hitl-two-group", job_id="job-hitl")
    state.agent_outputs = {
        "dataframe_json": df.to_json(),
        "variables": [
            {"name": "RECOVERY DAYS", "dtype": "numerical"},
            {"name": "RECOVERED", "dtype": "categorical"},
        ],
        "groups": {"column": "GROUP", "groups": ["Control", "Treatment"]},
        "study_design": "two_group",
        "sample_size": 60,
        "title": "Recovery After Fracture Treatment",
        "intent": "Compare recovery between control and treatment groups",
        "test_results": [
            {
                "variable_name": "RECOVERY DAYS",
                "dtype": "numerical",
                "descriptive": {"mean": 16.0, "std": 4.5, "min_val": 8, "max_val": 28},
                "frequency_table": {"rows": []},
                "test_result": {
                    "test_type": "unpaired_t_test",
                    "p_value": 0.02,
                    "is_significant_05": True,
                    "significance_statement": "Significant difference in recovery days (p=0.02)",
                },
            },
            {
                "variable_name": "RECOVERED",
                "dtype": "categorical",
                "descriptive": {},
                "frequency_table": {"rows": [{"category": "Yes", "count": 32, "percentage": 53.3}]},
                "test_result": {
                    "test_type": "chi_square",
                    "p_value": 0.08,
                    "is_significant_05": False,
                    "significance_statement": "No significant association for recovered (p=0.08)",
                },
            },
        ],
        "frequency_tables": [],
        "report_version": 1,
    }
    return state


def _methodology_paragraph(sections: list) -> str:
    for i, s in enumerate(sections):
        if s.get("type") == "heading" and "statistical analysis" in (s.get("content") or "").lower():
            if i + 1 < len(sections) and sections[i + 1].get("type") == "paragraph":
                return sections[i + 1].get("content", "")
    return ""


def _test_type_for_variable(test_results: list, variable: str) -> str | None:
    for tr in test_results:
        if tr.get("variable_name") == variable:
            inner = tr.get("test_result") or {}
            return inner.get("test_type")
    return None


@pytest.mark.asyncio
async def test_hitl_test_override_changes_inferential_test_and_methodology():
    """Professor requests Mann-Whitney instead of t-test — results and methodology must update."""
    state = _two_group_state()
    reporter = ReportGeneratorAgent(services=None)
    state = await reporter.run(state)

    state.agent_outputs["structured_review"] = {
        "test_overrides": [{"variable": "RECOVERY DAYS", "new_test": "mann_whitney_u"}],
        "section_feedback": [],
        "final_comment": "",
    }

    agent = RevisionResolverAgent(services=None)
    result = await agent.run(state)

    assert result.status != "FAILED", result.error_message

    test_type = _test_type_for_variable(result.agent_outputs["test_results"], "RECOVERY DAYS")
    assert test_type == "mann_whitney_u"

    methodology = _methodology_paragraph(result.agent_outputs["report_sections"])
    assert "Professor review note" not in methodology
    assert "Mann-Whitney" in methodology or "mann" in methodology.lower()

    assert result.agent_outputs["report_version"] == 2
    assert len(result.agent_outputs.get("charts", [])) >= 1


@pytest.mark.asyncio
async def test_hitl_executive_summary_and_methodology_feedback_without_inline_notes():
    """Executive summary and methodology presets must apply without pasting professor notes."""
    state = _two_group_state()
    reporter = ReportGeneratorAgent(services=None)
    state = await reporter.run(state)

    state.agent_outputs["structured_review"] = {
        "test_overrides": [{"variable": "RECOVERED", "new_test": "fisher_exact"}],
        "section_feedback": [
            {"target": "executive_summary", "preset_codes": ["add_limitations", "strengthen_conclusion"]},
            {"target": "methodology", "preset_codes": ["cite_assumptions"]},
        ],
        "final_comment": "Please emphasize clinical significance.",
    }

    agent = RevisionResolverAgent(services=None)
    result = await agent.run(state)

    assert result.status != "FAILED", result.error_message

    methodology = _methodology_paragraph(result.agent_outputs["report_sections"])
    assert "Professor review note" not in methodology
    assert "Professor comment" not in methodology

    recovered_test = _test_type_for_variable(result.agent_outputs["test_results"], "RECOVERED")
    assert recovered_test in ("fisher_exact", "chi_square")

    summary = json.loads(result.agent_outputs["summary"])
    assert "limitation" in summary.get("study_description", "").lower()
    assert "clinical significance" in summary.get("study_description", "").lower()


@pytest.mark.asyncio
async def test_hitl_variable_narrative_feedback_regenerates_report_sections():
    """Per-variable feedback should update report content via regeneration, not inline notes."""
    state = _two_group_state()
    reporter = ReportGeneratorAgent(services=None)
    state = await reporter.run(state)
    before_sections = len(state.agent_outputs["report_sections"])

    state.agent_outputs["structured_review"] = {
        "test_overrides": [],
        "section_feedback": [
            {"target": "RECOVERY DAYS", "preset_codes": ["rerun_descriptive"]},
        ],
        "final_comment": "",
    }

    agent = RevisionResolverAgent(services=None)
    result = await agent.run(state)

    assert result.status != "FAILED"
    assert len(result.agent_outputs["report_sections"]) >= before_sections

    for section in result.agent_outputs["report_sections"]:
        content = section.get("content") or ""
        assert "Professor review note" not in content


@pytest.mark.asyncio
async def test_hitl_methodology_wrong_test_auto_changes_inferential_test():
    """Methodology 'wrong test' feedback without explicit override still changes parametric tests."""
    state = _two_group_state()
    reporter = ReportGeneratorAgent(services=None)
    state = await reporter.run(state)

    state.agent_outputs["structured_review"] = {
        "test_overrides": [],
        "section_feedback": [
            {
                "target": "methodology",
                "preset_codes": ["wrong_test", "study_design_mismatch", "sample_size_low"],
            },
        ],
        "final_comment": "",
    }

    agent = RevisionResolverAgent(services=None)
    result = await agent.run(state)

    assert result.status != "FAILED", result.error_message
    test_type = _test_type_for_variable(result.agent_outputs["test_results"], "RECOVERY DAYS")
    assert test_type == "mann_whitney_u"

    methodology = _methodology_paragraph(result.agent_outputs["report_sections"])
    assert "Professor review note" not in methodology
    assert "Acknowledge sample size limitations" not in methodology


@pytest.mark.asyncio
async def test_hitl_methodology_only_presets_do_not_inject_raw_feedback_text():
    """Methodology chip feedback alone must not paste preset hint strings into the report."""
    state = _two_group_state()
    reporter = ReportGeneratorAgent(services=None)
    state = await reporter.run(state)

    state.agent_outputs["structured_review"] = {
        "test_overrides": [],
        "section_feedback": [
            {
                "target": "methodology",
                "preset_codes": ["sample_size_low", "multiple_testing", "study_design_mismatch"],
            },
        ],
        "final_comment": "",
    }

    agent = RevisionResolverAgent(services=None)
    result = await agent.run(state)

    assert result.status != "FAILED"
    methodology = _methodology_paragraph(result.agent_outputs["report_sections"])
    assert "Professor review note" not in methodology
    assert "Acknowledge sample size limitations" not in methodology
    assert "Statistical analysis was done" in methodology
