"""
AI Professor Review Agent — Automated quality review of generated analysis reports.

Three-tier validation:
  Tier 1: Mechanical/Clerical (deterministic, no LLM)
  Tier 2: Statistical Methodology (rule-based + heuristics)
  Tier 3: Clinical Interpretation (LLM-powered, flags for professor)

Produces a structured ReviewSummary that dramatically reduces professor review time.
"""
from typing import List, Optional
from io import StringIO

import pandas as pd

from src.agentic_core.agents.base import BaseAgent
from src.agentic_core.agents.state import MissionState
from src.agentic_core.agents.context import AgentContext, AgentPlan, AgentAction
from src.medstats.agents.review_agent.schemas import ReviewItem, ReviewSummary, Severity
from src.medstats.agents.review_agent.tier1_mechanical import run_tier1_checks
from src.medstats.agents.review_agent.tier2_methodology import run_tier2_checks
from src.medstats.agents.review_agent.tier3_interpretation import run_tier3_checks
from src.medstats.agents.rag_knowledge import get_rag_agent
from src.shared.logging.logger import get_logger

logger = get_logger(__name__)


class ReviewAgent(BaseAgent):
    """
    AI Professor Review Agent.
    Runs after ReportGeneratorAgent to validate the generated report.
    """

    role_name = "ReviewAgent"
    requires_llm_reasoning = False

    async def _perceive_domain(self, state: MissionState, context: AgentContext) -> AgentContext:
        """Gather report data for review."""
        df_json = state.agent_outputs.get("dataframe_json", "")
        if not df_json:
            parsed = state.agent_outputs.get("parsed_dataset")
            if parsed and hasattr(parsed, "dataframe"):
                df_json = parsed.dataframe.to_json()
            elif parsed and isinstance(parsed, str):
                df_json = parsed

        context.parsed_data = {
            "report_sections": state.agent_outputs.get("report_sections", []),
            "test_results": state.agent_outputs.get("test_results", []),
            "frequency_tables": state.agent_outputs.get("frequency_tables", []),
            "variables": state.agent_outputs.get("variables", []),
            "sample_size": state.agent_outputs.get("sample_size", 0),
            "study_design": state.agent_outputs.get("study_design", ""),
            "charts": state.agent_outputs.get("charts", []),
            "title": state.agent_outputs.get("title", ""),
            "dataframe_json": df_json,
        }
        return context

    async def _act_domain(self, state: MissionState, context: AgentContext, plan: AgentPlan):
        """Run all three tiers of review."""
        data = context.parsed_data
        items: List[ReviewItem] = []

        df = None
        if data["dataframe_json"]:
            try:
                df = pd.read_json(StringIO(data["dataframe_json"]))
            except Exception:
                pass

        items.extend(run_tier1_checks(data, df))
        items.extend(run_tier2_checks(data, df))
        items = await self._enrich_tier2_citations(items)

        tier3_items = await run_tier3_checks(data, df, services=self.services)
        items.extend(tier3_items)

        # Deduplicate: same check_id + variable should appear only once
        seen = set()
        deduped = []
        for item in items:
            key = (item.check_id, item.variable)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        items = deduped

        summary = self._build_summary(items)

        state.agent_outputs["review_summary"] = summary.to_dict()
        state.agent_outputs["review_verdict"] = summary.overall_verdict

        action = AgentAction.success_action(
            "review",
            output=(
                f"Review complete: {summary.passed} passed, "
                f"{summary.warnings} warnings, {summary.errors} errors. "
                f"Verdict: {summary.overall_verdict}"
            ),
        )
        return [action], state

    async def _enrich_tier2_citations(self, items: List[ReviewItem]) -> List[ReviewItem]:
        """Attach grounded citations from RAG to Tier 2 methodology flags."""
        import asyncio
        rag_agent = get_rag_agent()
        if not rag_agent:
            return items

        for item in items:
            if item.tier != 2 or item.severity not in (Severity.WARNING, Severity.ERROR):
                continue
            query = f"{item.category}: {item.message}. {item.suggestion}"
            tags = []
            if item.variable:
                tags.append(item.variable.lower().replace(" ", "_"))
            if "fisher" in item.suggestion.lower():
                tags.append("fisher_exact")
            if "mann" in item.suggestion.lower():
                tags.append("mann_whitney_u")

            try:
                result = await asyncio.wait_for(
                    rag_agent.retrieve(query, tags=tags or None, top_k=1),
                    timeout=10.0,
                )
                if result.citations:
                    c = result.citations[0]
                    item.citation = c.source + (f", p.{c.page}" if c.page else "")
                elif result.passages:
                    item.citation = result.passages[0][:250]
            except asyncio.TimeoutError:
                logger.warning("RAG citation enrichment timed out", check_id=item.check_id)
            except Exception as e:
                logger.warning("RAG citation enrichment failed", error=str(e))

        return items

    def _build_summary(self, items: List[ReviewItem]) -> ReviewSummary:
        """Build summary with verdict and estimated review time."""
        summary = ReviewSummary(
            total_checks=len(items),
            passed=sum(1 for i in items if i.severity == Severity.OK),
            info_count=sum(1 for i in items if i.severity == Severity.INFO),
            warnings=sum(1 for i in items if i.severity == Severity.WARNING),
            errors=sum(1 for i in items if i.severity == Severity.ERROR),
            items=items,
        )

        if summary.errors == 0 and summary.warnings <= 2:
            summary.estimated_review_minutes = 5
            summary.overall_verdict = "approve_recommended"
        elif summary.errors == 0 and summary.warnings <= 5:
            summary.estimated_review_minutes = 10
            summary.overall_verdict = "minor_issues"
        elif summary.errors <= 2:
            summary.estimated_review_minutes = 20
            summary.overall_verdict = "needs_attention"
        else:
            summary.estimated_review_minutes = 30
            summary.overall_verdict = "significant_issues"

        return summary
