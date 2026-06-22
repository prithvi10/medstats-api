"""
Integration tests using real MEDICINE 5 thesis data.
Tests the full pipeline: parse -> select tests -> run stats -> generate report.
"""
import pytest
import os
import pandas as pd
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def medicine5_excel_path():
    path = FIXTURES_DIR / "medicine5_data.xlsx"
    assert path.exists(), f"Fixture not found: {path}"
    return str(path)


@pytest.fixture
def medicine5_masterchart_path():
    path = FIXTURES_DIR / "MASTERCHART.docx"
    assert path.exists(), f"Fixture not found: {path}"
    return str(path)


@pytest.fixture
def medicine5_discussion_path():
    path = FIXTURES_DIR / "DISCUSSION.docx"
    assert path.exists(), f"Fixture not found: {path}"
    return str(path)


class TestDocumentParserIntegration:
    """Test document parsing with real MEDICINE 5 files."""

    def test_parse_excel_file(self, medicine5_excel_path):
        df = pd.read_excel(medicine5_excel_path, sheet_name=0, header=0)
        # Skip sub-header row
        if df.iloc[0].isna().sum() > len(df.columns) * 0.3:
            df = df.iloc[1:].reset_index(drop=True)

        assert len(df) >= 48  # ~50 patients (minus sub-header)
        assert "AGE" in df.columns or "AGE" in [c.upper() for c in df.columns]

    def test_parse_masterchart_docx_tables(self, medicine5_masterchart_path):
        from docx import Document
        doc = Document(medicine5_masterchart_path)
        assert len(doc.tables) == 2

        # Table 1: patient demographics
        table1 = doc.tables[0]
        headers = [cell.text.strip() for cell in table1.rows[0].cells]
        assert "AGE" in headers
        assert "SEX" in headers
        assert len(table1.rows) >= 50

        # Table 2: outcomes
        table2 = doc.tables[1]
        headers2 = [cell.text.strip() for cell in table2.rows[0].cells]
        assert any("OPERATIVE" in h.upper() for h in headers2)
        assert any("UNION" in h.upper() for h in headers2)

    @pytest.mark.asyncio
    async def test_full_parser_agent(self, medicine5_excel_path, medicine5_discussion_path):
        from src.medstats.agents.document_parser import DocumentParserAgent
        from src.agentic_core.agents.base_agent import MissionState

        agent = DocumentParserAgent(services=None)
        state = MissionState(submission_id="test-001", job_id="job-001")
        state.agent_outputs["files"] = [
            {"path": medicine5_excel_path, "type": "data_excel", "name": "medicine5_data.xlsx"},
            {"path": medicine5_discussion_path, "type": "supporting_doc", "name": "DISCUSSION.docx"},
        ]
        state.agent_outputs["intent"] = "Evaluate outcome of titanium elastic nail fixation for paediatric fractures"
        state.agent_outputs["title"] = "TENS for Paediatric Long Bone Fractures"

        result_state = await agent.run(state)

        assert result_state.status != "FAILED", f"Agent failed: {result_state.error_message}"
        assert "parsed_dataset" in result_state.agent_outputs
        assert result_state.agent_outputs["sample_size"] >= 48
        assert len(result_state.agent_outputs["variables"]) > 5

    @pytest.mark.asyncio
    async def test_parser_with_docx_tables(self, medicine5_masterchart_path, medicine5_discussion_path):
        from src.medstats.agents.document_parser import DocumentParserAgent
        from src.agentic_core.agents.base_agent import MissionState

        agent = DocumentParserAgent(services=None)
        state = MissionState(submission_id="test-002", job_id="job-002")
        state.agent_outputs["files"] = [
            {"path": medicine5_masterchart_path, "type": "data_excel", "name": "MASTERCHART.docx"},
            {"path": medicine5_discussion_path, "type": "supporting_doc", "name": "DISCUSSION.docx"},
        ]
        state.agent_outputs["intent"] = "Evaluate TENS for paediatric fractures"
        state.agent_outputs["title"] = "TENS Study"

        result_state = await agent.run(state)

        assert result_state.status != "FAILED", f"Agent failed: {result_state.error_message}"
        assert result_state.agent_outputs["sample_size"] >= 48


class TestTestSelectorIntegration:
    """Test statistical test selection with real parsed data."""

    @pytest.mark.asyncio
    async def test_full_test_selection_pipeline(self, medicine5_excel_path, medicine5_discussion_path):
        from src.medstats.agents.document_parser import DocumentParserAgent
        from src.medstats.agents.test_selector import TestSelectorAgent
        from src.agentic_core.agents.base_agent import MissionState

        # Step 1: Parse
        parser = DocumentParserAgent(services=None)
        state = MissionState(submission_id="test-003", job_id="job-003")
        state.agent_outputs["files"] = [
            {"path": medicine5_excel_path, "type": "data_excel", "name": "medicine5_data.xlsx"},
            {"path": medicine5_discussion_path, "type": "supporting_doc", "name": "DISCUSSION.docx"},
        ]
        state.agent_outputs["intent"] = "Statistical analysis of TENS outcomes"
        state.agent_outputs["title"] = "TENS Outcomes"

        state = await parser.run(state)
        assert state.status != "FAILED"

        # Step 2: Select and run tests
        selector = TestSelectorAgent(services=None)
        state = await selector.run(state)
        assert state.status != "FAILED"

        test_results = state.agent_outputs.get("test_results", [])
        assert len(test_results) > 0

        # Verify we got frequency tables
        freq_tables = state.agent_outputs.get("frequency_tables", [])
        assert len(freq_tables) > 0

        # Check that results contain expected variable types
        var_names = [r["variable_name"] for r in test_results if "variable_name" in r]
        assert len(var_names) > 3


class TestFullPipelineIntegration:
    """End-to-end pipeline test with real data."""

    @pytest.mark.asyncio
    async def test_complete_pipeline_medicine5(self, medicine5_excel_path, medicine5_discussion_path):
        from src.medstats.agents.document_parser import DocumentParserAgent
        from src.medstats.agents.test_selector import TestSelectorAgent
        from src.medstats.agents.report_generator import ReportGeneratorAgent
        from src.agentic_core.agents.base_agent import MissionState
        from src.agentic_core.agents.orchestrator import MissionControl

        state = MissionState(submission_id="test-e2e", job_id="job-e2e")
        state.agent_outputs["files"] = [
            {"path": medicine5_excel_path, "type": "data_excel", "name": "medicine5_data.xlsx"},
            {"path": medicine5_discussion_path, "type": "supporting_doc", "name": "DISCUSSION.docx"},
        ]
        state.agent_outputs["intent"] = "Evaluate outcome of TENS fixation for paediatric long bone fractures"
        state.agent_outputs["title"] = "Outcome of Titanium Elastic Nail Fixation for Paediatric Long Bone Fractures"

        agents = [
            DocumentParserAgent(services=None),
            TestSelectorAgent(services=None),
            ReportGeneratorAgent(services=None),  # No LLM — uses template fallback
        ]
        orchestrator = MissionControl(agents=agents)

        final_state = None
        async for updated_state in orchestrator.execute(state):
            final_state = updated_state

        assert final_state is not None
        assert final_state.status == "COMPLETED", f"Pipeline failed: {final_state.error_message}"
        assert final_state.progress == 100

        # Verify report was generated
        sections = final_state.agent_outputs.get("report_sections", [])
        assert len(sections) > 5

        # Verify charts were generated
        charts = final_state.agent_outputs.get("charts", [])
        assert len(charts) > 0
        for chart in charts:
            assert os.path.exists(chart["path"]), f"Chart file missing: {chart['path']}"

        # Verify methodology note
        methodology = final_state.agent_outputs.get("methodology_note", "")
        assert "Statistical analysis" in methodology

        # Verify summary
        summary = final_state.agent_outputs.get("summary", "")
        assert "Sample size" in summary

        # Print report for manual inspection
        print("\n" + "=" * 80)
        print("GENERATED REPORT PREVIEW")
        print("=" * 80)
        for section in sections[:15]:
            if section["type"] == "heading":
                print(f"\n{'='*40}")
                print(f"  {section['content']}")
                print(f"{'='*40}")
            elif section["type"] == "table_heading":
                print(f"\n--- {section['content']} ---")
            elif section["type"] == "paragraph":
                print(f"  {section['content'][:200]}")
            elif section["type"] == "test_result":
                print(f"  >> {section['content']}")
            elif section["type"] == "table":
                headers = section.get("headers", [])
                rows = section.get("rows", [])[:5]
                print(f"  | {' | '.join(headers)} |")
                for row in rows:
                    print(f"  | {' | '.join(row)} |")
        print(f"\n  ... ({len(sections)} total sections, {len(charts)} charts)")
        print("=" * 80)
