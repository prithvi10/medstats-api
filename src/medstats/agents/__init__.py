"""
MedStats Agents — Public API.

All agents follow the 4-phase agentic loop (Perceive → Reason → Act → Feedback).
Each agent is a directory with: agent.py, schemas.py, prompts.py, and optional helpers.

Usage:
    from src.medstats.agents import (
        DocumentParserAgent,
        TestSelectorAgent,
        ReportGeneratorAgent,
        ReviewAgent,
        CorrectionAgent,
    )
"""
from src.medstats.agents.document_parser import DocumentParserAgent  # noqa: F401
from src.medstats.agents.test_selector import TestSelectorAgent  # noqa: F401
from src.medstats.agents.report_generator import ReportGeneratorAgent  # noqa: F401
from src.medstats.agents.review_agent import ReviewAgent  # noqa: F401
from src.medstats.agents.correction_agent import CorrectionAgent  # noqa: F401
