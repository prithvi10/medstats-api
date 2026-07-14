#!/usr/bin/env python3
"""
generate_sample_reports.py — Run the full agentic pipeline on 5 real thesis datasets.

Generates DOCX reports that can be compared against manually-generated results.
Output goes to: generated_reports/comparisons/
"""
import asyncio
import sys
import os
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.medstats.agents.document_parser import DocumentParserAgent
from src.medstats.agents.test_selector import TestSelectorAgent
from src.medstats.agents.report_generator import ReportGeneratorAgent
from src.medstats.agents.review_agent import ReviewAgent
from src.medstats.services.export_service import export_report_to_docx
from src.agentic_core.agents.base_agent import MissionState
from src.agentic_core.llm.service import get_llm_service

DATA_ROOT = Path("/Users/prithviraj/Thesis Project/data")
OUTPUT_DIR = Path(__file__).parent.parent / "generated_reports" / "comparisons"

# 5 datasets with their input Excel and manual Results doc for comparison
DATASETS = [
    {
        "id": "MEDICINE_5",
        "name": "TENS for Paediatric Long Bone Fractures",
        "input_excel": DATA_ROOT / "MEDICINE/MEDICINE 5/New Microsoft Office Excel Worksheet.xlsx",
        "supporting_doc": DATA_ROOT / "MEDICINE/MEDICINE 5/DISCUSSION.docx",
        "manual_results": DATA_ROOT / "MEDICINE/MEDICINE 5/Results.docx",
        "intent": "Evaluate clinical and radiological outcome of titanium elastic nail fixation for paediatric long bone fractures",
    },
    {
        "id": "MEDICINE_3",
        "name": "Distal Tibia Fractures",
        "input_excel": DATA_ROOT / "MEDICINE/MEDICINE 3/MASTER CHART PT FINAL CORRECTION.xlsx",
        "manual_results": DATA_ROOT / "MEDICINE/MEDICINE 3/NEW Results .docx",
        "intent": "Analyze outcomes of distal tibia fracture treatment methods",
    },
    {
        "id": "MEDICINE_4",
        "name": "Caesarean Section Analysis",
        "input_excel": DATA_ROOT / "MEDICINE/MEDICINE 4/MASTER CHART final.xlsx",
        "supporting_doc": DATA_ROOT / "MEDICINE/MEDICINE 4/observation and result.docx",
        "manual_results": DATA_ROOT / "MEDICINE/MEDICINE 4/observation and result.docx",
        "intent": "Analyze indications and outcomes of caesarean section deliveries",
    },
    {
        "id": "DENTISTRY_1",
        "name": "Dental Statistics Study",
        "input_excel": DATA_ROOT / "DENTISTRY/DENTISTRY 1/STATISTICS - NEW .xlsx",
        "manual_results": DATA_ROOT / "DENTISTRY/DENTISTRY 1/Final RESULTS .docx",
        "intent": "Statistical analysis of dental treatment outcomes and measurements",
    },
    {
        "id": "MEDICINE_6",
        "name": "Medical Observations Study",
        "input_excel": DATA_ROOT / "MEDICINE/MEDICINE 6/15 Masterchart.xlsx",
        "supporting_doc": DATA_ROOT / "MEDICINE/MEDICINE 6/07 MATERIAL AND METHODS.docx",
        "manual_results": DATA_ROOT / "MEDICINE/MEDICINE 6/Results .docx",
        "intent": "Analyze clinical observations and patient outcomes from masterchart data",
    },
    {
        "id": "MEDICINE_2",
        "name": "MACS Project Clinical Study",
        "input_excel": DATA_ROOT / "MEDICINE/MEDICINE 2/macs project NEW.xlsx",
        "manual_results": DATA_ROOT / "MEDICINE/MEDICINE 2/Results.docx",
        "intent": "Analyze clinical outcomes and patient characteristics from MACS project data",
    },
]


def print_section(title: str):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def print_review_summary(summary: dict):
    print(f"  AI Review: {summary['passed']} passed | "
          f"{summary['info_count']} info | "
          f"{summary['warnings']} warnings | "
          f"{summary['errors']} errors")
    print(f"  Verdict: {summary['overall_verdict']} "
          f"(est. {summary['estimated_review_minutes']} min professor review)")
    if summary.get("items"):
        print(f"  Flagged items:")
        for item in summary["items"][:5]:
            icon = {"info": "ℹ️", "warning": "⚠️", "error": "❌"}.get(item["severity"], "•")
            print(f"    {icon} [{item['category']}] {item['message']}")


async def run_pipeline(dataset: dict) -> dict:
    """Run the full pipeline on a single dataset."""
    ds_id = dataset["id"]
    name = dataset["name"]
    excel_path = dataset["input_excel"]

    # Get LLM service (None if no API key configured)
    llm_service = get_llm_service()

    print_section(f"{ds_id}: {name}")
    if llm_service:
        print(f"  LLM: GPT-5 enabled")
    else:
        print(f"  LLM: Disabled (no API key) — using templates")

    if not excel_path.exists():
        print(f"  ⚠️ SKIPPED - Input file not found: {excel_path}")
        return {"id": ds_id, "status": "skipped", "reason": "file not found"}

    print(f"  Input: {excel_path.name}")
    print(f"  Intent: {dataset['intent']}")

    # Build initial state
    state = MissionState(submission_id=ds_id, job_id=f"report-{ds_id}")
    files = [{"path": str(excel_path), "type": "data_excel", "name": excel_path.name}]

    if dataset.get("supporting_doc") and Path(dataset["supporting_doc"]).exists():
        files.append({
            "path": str(dataset["supporting_doc"]),
            "type": "supporting_doc",
            "name": Path(dataset["supporting_doc"]).name,
        })

    state.agent_outputs = {
        "files": files,
        "intent": dataset["intent"],
        "title": name,
    }

    # Phase 1: Document Parser
    print(f"\n  [1/4] Parsing documents...")
    parser = DocumentParserAgent(services=None)
    state = await parser.run(state)
    if state.status == "FAILED":
        print(f"  ❌ Parser FAILED: {state.error_message}")
        return {"id": ds_id, "status": "failed", "stage": "parser", "error": state.error_message}

    n_vars = len(state.agent_outputs.get("variables", []))
    sample_n = state.agent_outputs.get("sample_size", 0)
    print(f"       ✓ Parsed {n_vars} variables, N={sample_n}")

    # Phase 2: Test Selector
    print(f"  [2/4] Selecting & running statistical tests...")
    selector = TestSelectorAgent(services=None)
    state = await selector.run(state)
    if state.status == "FAILED":
        print(f"  ❌ TestSelector FAILED: {state.error_message}")
        return {"id": ds_id, "status": "failed", "stage": "selector", "error": state.error_message}

    tests_count = state.agent_outputs.get("tests_run_count", 0)
    print(f"       ✓ Ran {tests_count} statistical tests")

    # Phase 3: Report Generator (uses LLM for narrative)
    print(f"  [3/4] Generating report...")
    reporter = ReportGeneratorAgent(services=llm_service)
    state = await reporter.run(state)
    if state.status == "FAILED":
        print(f"  ❌ ReportGenerator FAILED: {state.error_message}")
        print(f"       Attempting partial export with available data...")

    sections = state.agent_outputs.get("report_sections", [])
    charts = state.agent_outputs.get("charts", [])
    print(f"       {'✓' if state.status != 'FAILED' else '~'} "
          f"{len(sections)} sections, {len(charts)} charts")

    # Phase 4: AI Review (uses LLM for Tier 3 interpretation)
    print(f"  [4/4] Running AI Professor Review...")
    reviewer = ReviewAgent(services=llm_service)
    state = await reviewer.run(state)
    review = state.agent_outputs.get("review_summary", {})
    if review:
        print_review_summary(review)

    # Export to DOCX
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = None

    if sections:
        try:
            output_path = export_report_to_docx(
                report_sections=sections,
                charts=charts,
                title=name,
                methodology_note=state.agent_outputs.get("methodology_note", ""),
                summary=state.agent_outputs.get("summary", ""),
                submission_id=ds_id,
            )
            # Move to comparisons folder
            dest = OUTPUT_DIR / f"{ds_id}_AI_GENERATED.docx"
            Path(output_path).rename(dest)
            output_path = str(dest)
            print(f"\n  📄 Report saved: {dest.name}")
        except Exception as e:
            print(f"\n  ⚠️ Export failed: {e}")
    else:
        print(f"\n  ⚠️ No sections to export")

    # Copy manual results for easy comparison
    manual = dataset.get("manual_results")
    if manual and Path(manual).exists():
        import shutil
        manual_dest = OUTPUT_DIR / f"{ds_id}_MANUAL.docx"
        if not manual_dest.exists():
            shutil.copy2(manual, manual_dest)
        print(f"  📄 Manual results: {manual_dest.name}")

    # Print test results summary
    test_results = state.agent_outputs.get("test_results", [])
    sig_tests = []
    for tr in test_results:
        inner = tr.get("test_result") or tr
        if inner and isinstance(inner, dict):
            p = inner.get("p_value")
            if p is not None and isinstance(p, (int, float)) and p < 0.05:
                sig_tests.append({
                    "var": tr.get("variable_name", inner.get("variable_name", "?")),
                    "test": inner.get("test_type", "?"),
                    "p": p,
                })

    if sig_tests:
        print(f"\n  Significant findings ({len(sig_tests)}):")
        for st in sig_tests[:8]:
            print(f"    • {st['var']}: {st['test']} (p={st['p']:.4f})")

    return {
        "id": ds_id,
        "status": "success" if state.status != "FAILED" else "partial",
        "output": output_path,
        "n_vars": n_vars,
        "sample_size": sample_n,
        "tests_run": tests_count,
        "sections": len(sections),
        "review_verdict": review.get("overall_verdict", "N/A"),
    }


async def main():
    print_section("MedStats AI — Sample Report Generation")
    print(f"  Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Datasets: {len(DATASETS)}")

    results = []
    for dataset in DATASETS:
        try:
            result = await run_pipeline(dataset)
            results.append(result)
        except Exception as e:
            print(f"\n  💥 UNEXPECTED ERROR in {dataset['id']}: {e}")
            import traceback
            traceback.print_exc()
            results.append({"id": dataset["id"], "status": "error", "error": str(e)})

    # Final summary
    print_section("SUMMARY")
    print(f"  {'ID':<15} {'Status':<10} {'N':<6} {'Vars':<6} {'Tests':<6} {'Verdict'}")
    print(f"  {'-'*15} {'-'*10} {'-'*6} {'-'*6} {'-'*6} {'-'*20}")
    for r in results:
        print(f"  {r['id']:<15} {r.get('status','?'):<10} "
              f"{r.get('sample_size','?'):<6} {r.get('n_vars','?'):<6} "
              f"{r.get('tests_run','?'):<6} {r.get('review_verdict','N/A')}")

    successful = [r for r in results if r["status"] in ("success", "partial")]
    print(f"\n  ✓ {len(successful)}/{len(DATASETS)} reports generated")
    print(f"  📁 Open folder: {OUTPUT_DIR}")
    print(f"\n  Compare AI vs Manual reports side-by-side in the output folder.")


if __name__ == "__main__":
    asyncio.run(main())
