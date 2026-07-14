"""Run a single student scenario through the analysis pipeline."""
from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

from src.agentic_core.agents.base_agent import MissionState
from src.agentic_core.agents.orchestrator import MissionControl
from src.medstats.agents.document_parser import DocumentParserAgent
from src.medstats.agents.report_generator import ReportGeneratorAgent
from src.medstats.agents.test_selector import TestSelectorAgent
from tests.scenarios.definitions import StudentScenario


async def run_scenario_pipeline(
    scenario: StudentScenario,
    *,
    chart_dir: str,
) -> Tuple[MissionState, Dict[str, Any]]:
    import src.medstats.core.chart_generator as cg

    prev_dir = cg.CHART_DIR
    cg.CHART_DIR = chart_dir
    os.makedirs(chart_dir, exist_ok=True)

    data_path = str(scenario.data_path())
    intent_path = str(scenario.intent_path())

    state = MissionState(submission_id=f"scenario-{scenario.id}", job_id=f"job-{scenario.id}")
    state.agent_outputs["files"] = [
        {"path": data_path, "type": "data_excel", "name": os.path.basename(data_path)},
        {"path": intent_path, "type": "supporting_doc", "name": os.path.basename(intent_path)},
    ]
    state.agent_outputs["intent"] = scenario.intent
    state.agent_outputs["title"] = scenario.title

    agents = [
        DocumentParserAgent(services=None),
        TestSelectorAgent(services=None),
        ReportGeneratorAgent(services=None),
    ]
    orchestrator = MissionControl(agents=agents)

    final_state = None
    async for updated in orchestrator.execute(state):
        final_state = updated

    cg.CHART_DIR = prev_dir

    if final_state is None:
        raise RuntimeError("Pipeline produced no state")

    meta = {
        "study_design": final_state.agent_outputs.get("study_design"),
        "sample_size": final_state.agent_outputs.get("sample_size", 0),
        "variables": final_state.agent_outputs.get("variables", []),
        "test_results": final_state.agent_outputs.get("test_results", []),
        "sections": final_state.agent_outputs.get("report_sections", []),
        "charts": final_state.agent_outputs.get("charts", []),
        "summary": final_state.agent_outputs.get("summary", ""),
        "methodology_note": final_state.agent_outputs.get("methodology_note", ""),
    }
    return final_state, meta


def summarize_scenario_result(scenario: StudentScenario, state: MissionState, meta: Dict[str, Any]) -> List[str]:
    lines = [
        f"Scenario: {scenario.id}",
        f"Status: {state.status}",
        f"Study design: {meta.get('study_design')} (expected {scenario.study_design})",
        f"Sample size: {meta.get('sample_size')}",
        f"Variables: {len(meta.get('variables', []))}",
        f"Charts: {len(meta.get('charts', []))}",
        f"Report sections: {len(meta.get('sections', []))}",
    ]
    if state.error_message:
        lines.append(f"Error: {state.error_message}")
    return lines
