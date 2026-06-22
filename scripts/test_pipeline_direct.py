#!/usr/bin/env python3
"""
Run the analysis pipeline directly on sample data without needing any infra.
No database, no Redis, no API server. Pure pipeline test.

Usage:
    python scripts/test_pipeline_direct.py [path_to_excel] [path_to_discussion]

Defaults to MEDICINE 5 fixtures if no arguments provided.
"""
import sys
import os
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.medstats.agents.document_parser import DocumentParserAgent
from src.medstats.agents.test_selector import TestSelectorAgent
from src.medstats.agents.report_generator import ReportGeneratorAgent
from src.agentic_core.agents.base_agent import MissionState
from src.agentic_core.agents.orchestrator import MissionControl


FIXTURES_DIR = Path(__file__).parent.parent / "tests" / "fixtures"


async def run_pipeline(data_path: str, discussion_path: str, title: str = ""):
    print("=" * 70)
    print("  MEDSTATS AI - DIRECT PIPELINE TEST")
    print("=" * 70)
    print(f"\n  Data file:       {data_path}")
    print(f"  Discussion file: {discussion_path}")
    print(f"  Title:           {title or '(auto-detect)'}")
    print()

    state = MissionState(submission_id="direct-test", job_id="direct-job")
    state.agent_outputs["files"] = [
        {"path": data_path, "type": "data_excel", "name": os.path.basename(data_path)},
        {"path": discussion_path, "type": "supporting_doc", "name": os.path.basename(discussion_path)},
    ]
    state.agent_outputs["intent"] = "Perform comprehensive statistical analysis"
    state.agent_outputs["title"] = title or "Statistical Analysis"

    agents = [
        DocumentParserAgent(services=None),
        TestSelectorAgent(services=None),
        ReportGeneratorAgent(services=None),
    ]
    orchestrator = MissionControl(agents=agents)

    print("  Running pipeline...\n")
    agent_names = ["DocumentParser", "TestSelector", "ReportGenerator"]

    final_state = None
    async for updated_state in orchestrator.execute(state):
        final_state = updated_state
        idx = updated_state.current_agent_index
        agent_name = agent_names[min(idx, len(agent_names) - 1)]
        print(f"  [{updated_state.progress:3d}%] {agent_name}: {updated_state.status}")

    print()

    if final_state.status == "FAILED":
        print(f"  PIPELINE FAILED: {final_state.error_message}")
        return 1

    # Print summary
    print("=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n  Sample size: {final_state.agent_outputs.get('sample_size', 'N/A')}")
    print(f"  Variables detected: {len(final_state.agent_outputs.get('variables', []))}")
    print(f"  Tests run: {len(final_state.agent_outputs.get('test_results', []))}")
    print(f"  Charts generated: {len(final_state.agent_outputs.get('charts', []))}")
    print(f"  Report sections: {len(final_state.agent_outputs.get('report_sections', []))}")

    # Variables
    variables = final_state.agent_outputs.get("variables", [])
    print(f"\n  Variables:")
    for v in variables:
        print(f"    - {v['name']} ({v['dtype']})")

    # Test results summary
    test_results = final_state.agent_outputs.get("test_results", [])
    print(f"\n  Statistical Tests:")
    for tr in test_results:
        inner = tr.get("test_result", {})
        if not inner:
            continue
        var = tr.get("variable_name", "N/A")
        sig = "***" if inner.get("is_significant_01") else ("**" if inner.get("is_significant_05") else "ns")
        p_val = inner.get("p_value")
        p_str = f"{p_val:.4f}" if isinstance(p_val, (int, float)) else str(p_val)
        print(f"    - {var}: {inner.get('test_type', 'N/A')} (p={p_str}) [{sig}]")

    # Charts
    charts = final_state.agent_outputs.get("charts", [])
    print(f"\n  Charts generated:")
    for c in charts:
        exists = "OK" if os.path.exists(c["path"]) else "MISSING"
        label = c.get("title") or c.get("variable", "chart")
        print(f"    - [{exists}] {label}: {c['path']}")

    # Report preview
    sections = final_state.agent_outputs.get("report_sections", [])
    print(f"\n  Report Preview (first 20 sections):")
    print("  " + "-" * 66)
    for section in sections[:20]:
        if section["type"] == "heading":
            print(f"\n  {'='*50}")
            print(f"  {section['content']}")
            print(f"  {'='*50}")
        elif section["type"] == "table_heading":
            print(f"\n  --- {section['content']} ---")
        elif section["type"] == "paragraph":
            content = section["content"]
            if len(content) > 120:
                content = content[:120] + "..."
            print(f"  {content}")
        elif section["type"] == "test_result":
            print(f"  >> {section['content']}")
        elif section["type"] == "table":
            headers = section.get("headers", [])
            rows = section.get("rows", [])[:3]
            print(f"  | {' | '.join(headers)} |")
            for row in rows:
                print(f"  | {' | '.join(str(c) for c in row)} |")
            if len(section.get("rows", [])) > 3:
                print(f"  | ... ({len(section['rows'])} total rows) |")

    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE - ALL STAGES PASSED")
    print("=" * 70)
    return 0


def main():
    if len(sys.argv) >= 3:
        data_path = sys.argv[1]
        discussion_path = sys.argv[2]
        title = sys.argv[3] if len(sys.argv) > 3 else ""
    else:
        data_path = str(FIXTURES_DIR / "medicine5_data.xlsx")
        discussion_path = str(FIXTURES_DIR / "DISCUSSION.docx")
        title = "Outcome of Titanium Elastic Nail Fixation for Paediatric Long Bone Fractures"

    if not os.path.exists(data_path):
        print(f"Error: Data file not found: {data_path}")
        sys.exit(1)
    if not os.path.exists(discussion_path):
        print(f"Error: Discussion file not found: {discussion_path}")
        sys.exit(1)

    exit_code = asyncio.run(run_pipeline(data_path, discussion_path, title))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
