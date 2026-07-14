"""Unit tests for the DOCX and PDF export service."""
import json
import pytest
import os
from pathlib import Path

from src.medstats.services.export_service import (
    build_export_document_title,
    build_export_filename,
    generate_preview_html,
)

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


class TestExportTitles:
    def test_document_title_suffix(self):
        assert build_export_document_title("TENS Study") == "TENS Study - Statistical analysis"
        assert build_export_document_title("TENS Study - Statistical analysis") == "TENS Study - Statistical analysis"

    def test_filename(self):
        assert build_export_filename("TENS Study", "docx") == "TENS Study - Statistical analysis.docx"


class TestPreviewHtmlOrder:
    def test_executive_summary_before_observations(self, sample_report_data):
        html = generate_preview_html(
            report_sections=sample_report_data["report_sections"],
            charts=sample_report_data["charts"],
            title=sample_report_data["title"],
            methodology_note=sample_report_data["methodology_note"],
            summary=sample_report_data["summary"],
        )
        exec_idx = html.index("Executive Summary")
        obs_idx = html.index("OBSERVATIONS AND RESULTS")
        method_idx = html.index("Statistical Methodology")
        assert exec_idx < obs_idx < method_idx
        assert "TENS Study - Statistical analysis" in html or sample_report_data["title"] in html

    def test_structured_executive_summary_formatting(self, sample_report_data):
        structured = json.dumps({
            "study_description": "A study of 50 patients with long bone fractures.",
            "methodology": "Chi-square and Z-tests were applied.",
            "key_findings": [
                {"variable": "Sex", "result": "Male predominance (p<0.001)", "significant": True},
                {"variable": "Age", "result": "No significant difference (p=0.23)", "significant": False},
            ],
            "stats": {"total_variables": 2, "significant": 1, "not_significant": 1},
        })
        html = generate_preview_html(
            report_sections=sample_report_data["report_sections"],
            charts=[],
            title=sample_report_data["title"],
            summary=structured,
        )
        assert "Study Description" in html
        assert "Key Findings" in html
        assert "finding-item sig" in html
        assert "variables analyzed" in html


@pytest.fixture
def sample_report_data():
    """Minimal report data matching pipeline output structure."""
    return {
        "report_sections": [
            {"type": "heading", "content": "OBSERVATIONS AND RESULTS"},
            {"type": "paragraph", "content": "A study was conducted on 50 subjects."},
            {"type": "table_heading", "content": "Table No.1: AGE"},
            {"type": "paragraph", "content": "The age ranged from 6 to 15 with mean 10.15 and SD 2.83."},
            {"type": "table", "headers": ["Category", "Number", "Percentage"],
             "rows": [["5-7", "8", "16%"], ["7-9", "6", "12%"], ["9-11", "20", "40%"],
                      ["11-13", "10", "20%"], ["13-15", "6", "12%"], ["TOTAL", "50", "100%"]]},
            {"type": "test_result", "content": "Age distribution was not statistically significant (p=0.23)."},
            {"type": "table_heading", "content": "Table No.2: SEX"},
            {"type": "paragraph", "content": "Among 50 subjects, majority (84%) were male."},
            {"type": "table", "headers": ["Category", "Number", "Percentage"],
             "rows": [["Male", "42", "84%"], ["Female", "8", "16%"], ["TOTAL", "50", "100%"]]},
            {"type": "test_result", "content": "Z-test shows male predominance is significant (Z=4.81, p<0.001)."},
            {"type": "heading", "content": "Statistical Analysis"},
            {"type": "paragraph", "content": "Statistical analysis was performed using Z-test and Chi-square test."},
        ],
        "charts": [],
        "title": "Outcome of TENS for Paediatric Long Bone Fractures",
        "methodology_note": "Statistical analysis was performed using SPSS v26.",
        "summary": "Sample size: 50. Key finding: Male predominance.",
    }


class TestDocxExport:
    def test_generates_docx_file(self, sample_report_data, tmp_path):
        from src.medstats.services.export_service import export_report_to_docx

        filepath = export_report_to_docx(
            report_sections=sample_report_data["report_sections"],
            charts=sample_report_data["charts"],
            title=sample_report_data["title"],
            methodology_note=sample_report_data["methodology_note"],
            summary=sample_report_data["summary"],
            submission_id="test-export-001",
            output_dir=str(tmp_path),
        )

        assert os.path.exists(filepath)
        assert filepath.endswith(".docx")
        assert os.path.getsize(filepath) > 1000

        from docx import Document
        doc = Document(filepath)

        headings = [p.text for p in doc.paragraphs if p.style.name.startswith("Heading")]
        assert "Executive Summary" in headings
        assert "OBSERVATIONS AND RESULTS" in headings
        assert "Statistical Methodology" in headings
        assert headings.index("Executive Summary") < headings.index("OBSERVATIONS AND RESULTS")

        assert len(doc.tables) >= 2

        table1 = doc.tables[0]
        headers = [cell.text for cell in table1.rows[0].cells]
        assert "Category" in headers
        assert "Number" in headers

    def test_with_real_pipeline_output(self, tmp_path):
        """Run the pipeline on real data and export."""
        import asyncio
        from src.medstats.agents.document_parser import DocumentParserAgent
        from src.medstats.agents.test_selector import TestSelectorAgent
        from src.medstats.agents.report_generator import ReportGeneratorAgent
        from src.agentic_core.agents.base_agent import MissionState
        from src.agentic_core.agents.orchestrator import MissionControl
        from src.medstats.services.export_service import export_report_to_docx

        excel_path = str(FIXTURES_DIR / "medicine5_data.xlsx")
        discussion_path = str(FIXTURES_DIR / "DISCUSSION.docx")

        async def run():
            state = MissionState(submission_id="export-test", job_id="export-job")
            state.agent_outputs["files"] = [
                {"path": excel_path, "type": "data_excel", "name": "medicine5_data.xlsx"},
                {"path": discussion_path, "type": "supporting_doc", "name": "DISCUSSION.docx"},
            ]
            state.agent_outputs["intent"] = "Evaluate TENS for paediatric fractures"
            state.agent_outputs["title"] = "TENS for Paediatric Long Bone Fractures"

            agents = [
                DocumentParserAgent(services=None),
                TestSelectorAgent(services=None),
                ReportGeneratorAgent(services=None),
            ]
            orchestrator = MissionControl(agents=agents)
            final = None
            async for s in orchestrator.execute(state):
                final = s
            return final

        final_state = asyncio.run(run())
        assert final_state.status == "COMPLETED"

        filepath = export_report_to_docx(
            report_sections=final_state.agent_outputs["report_sections"],
            charts=final_state.agent_outputs["charts"],
            title=final_state.agent_outputs.get("title", "TENS Study"),
            methodology_note=final_state.agent_outputs.get("methodology_note", ""),
            summary=final_state.agent_outputs.get("summary", ""),
            submission_id="medicine5-full",
            output_dir=str(tmp_path),
        )

        assert os.path.exists(filepath)
        size = os.path.getsize(filepath)
        assert size > 5000

        from docx import Document
        doc = Document(filepath)
        assert len(doc.tables) >= 5
        assert len(doc.paragraphs) > 20

        print(f"\n  Exported DOCX: {filepath} ({size / 1024:.1f} KB)")
        print(f"  Tables: {len(doc.tables)}, Paragraphs: {len(doc.paragraphs)}")


class TestPdfExport:
    def test_generates_pdf_file(self, sample_report_data, tmp_path):
        from src.medstats.services.export_service import export_report_to_pdf

        filepath = export_report_to_pdf(
            report_sections=sample_report_data["report_sections"],
            charts=sample_report_data["charts"],
            title=sample_report_data["title"],
            methodology_note=sample_report_data["methodology_note"],
            summary=sample_report_data["summary"],
            submission_id="test-pdf-001",
            output_dir=str(tmp_path),
        )

        assert os.path.exists(filepath)
        assert filepath.endswith(".pdf")
        assert os.path.getsize(filepath) > 1000
