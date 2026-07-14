"""
LiteratureMatchAgent — matches study findings to comparable published literature.

Runs after ReportGeneratorAgent. Output: literature_matches + optional report section.
"""
from __future__ import annotations

import asyncio
import re
from typing import Any, Dict, List

from src.agentic_core.agents.base import BaseAgent
from src.agentic_core.agents.state import MissionState
from src.agentic_core.agents.context import AgentContext, AgentPlan, AgentAction
from src.medstats.rag.embedder import ChunkEmbedder
from src.medstats.rag.literature_search import LiteratureSearchEngine
from src.shared.config.settings import get_settings
from src.shared.db.session import async_session_factory
from src.shared.logging.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


class LiteratureMatchAgent(BaseAgent):
    role_name = "LiteratureMatch"
    requires_llm_reasoning = False

    async def _perceive_domain(self, state: MissionState, context: AgentContext) -> AgentContext:
        context.parsed_data = {
            "title": state.agent_outputs.get("title", ""),
            "study_design": state.agent_outputs.get("study_design", ""),
            "sample_size": state.agent_outputs.get("sample_size", 0),
            "test_results": state.agent_outputs.get("test_results", []),
            "variables": state.agent_outputs.get("variables", []),
            "intent": state.agent_outputs.get("intent", ""),
            "report_sections": state.agent_outputs.get("report_sections", []),
        }
        return context

    async def _act_domain(self, state: MissionState, context: AgentContext, plan: AgentPlan):
        data = context.parsed_data
        query_text = self._build_query(data)
        disciplines = self._infer_disciplines(data)
        methods_tags = self._extract_methods_tags(data)

        matches = await self._search_literature(query_text, disciplines, methods_tags)

        state.agent_outputs["literature_matches"] = [
            {
                "pmid": m.pmid,
                "title": m.title,
                "authors": m.authors,
                "year": m.year,
                "journal": m.journal,
                "abstract_snippet": m.abstract_snippet,
                "similarity_score": m.similarity_score,
            }
            for m in matches
        ]

        if matches and data.get("report_sections") is not None:
            section_content = self._format_literature_section(matches)
            sections = list(data["report_sections"])
            sections.append({"type": "heading", "content": "Prior Literature Context"})
            sections.append({"type": "literature_context", "content": section_content})
            state.agent_outputs["report_sections"] = sections

        action = AgentAction.success_action(
            "literature_match",
            output=f"Found {len(matches)} comparable studies",
        )
        return [action], state

    def _build_query(self, data: dict) -> str:
        title = data.get("title", "")
        intent = data.get("intent", "")
        design = data.get("study_design", "")
        n = data.get("sample_size", 0)

        significant = []
        for tr in data.get("test_results", []):
            td = tr.get("test_result")
            if td and isinstance(td, dict) and td.get("is_significant_05"):
                significant.append(tr.get("variable_name", ""))

        sig_text = ", ".join(significant[:5]) if significant else "statistical outcomes"
        return (
            f"{title}. {intent}. {design} study N={n}. "
            f"Significant findings: {sig_text}. comparable published studies"
        )

    def _infer_disciplines(self, data: dict) -> List[str]:
        text = (data.get("title", "") + " " + data.get("intent", "")).lower()
        mapping = {
            "ortho": "orthopaedics",
            "fracture": "orthopaedics",
            "dental": "dentistry",
            "physio": "physiotherapy",
            "eye": "ophthalmology",
            "obstetric": "obstetrics_gynaecology",
            "paediatric": "paediatrics",
            "pediatric": "paediatrics",
            "psychiatr": "psychiatry",
            "ent": "ent_surgery",
            "nursing": "nursing",
            "pharmacy": "pharmacy",
            "ayurved": "ayurveda",
            "community": "community_medicine",
        }
        found = []
        for keyword, discipline in mapping.items():
            if re.search(rf"\b{re.escape(keyword)}\b", text):
                found.append(discipline)
        return found or ["general_medicine"]

    def _extract_methods_tags(self, data: dict) -> List[str]:
        tags = set()
        for tr in data.get("test_results", []):
            td = tr.get("test_result")
            if td and isinstance(td, dict) and td.get("test_type"):
                tags.add(td["test_type"])
        n = data.get("sample_size", 0)
        if n and n < 100:
            tags.add("n_lt_100")
        return list(tags)

    async def _search_literature(
        self,
        query: str,
        disciplines: List[str],
        methods_tags: List[str],
    ) -> List:
        if not settings.OPENAI_API_KEY or "sqlite" in settings.DATABASE_URL:
            return []

        try:
            embedder = ChunkEmbedder()
            embedding = await asyncio.wait_for(
                embedder.embed_query(query), timeout=30.0
            )
            async with async_session_factory() as session:
                engine = LiteratureSearchEngine(session)
                return await asyncio.wait_for(
                    engine.search(
                        query_embedding=embedding,
                        disciplines=disciplines,
                        methods_tags=methods_tags,
                        limit=3,
                    ),
                    timeout=15.0,
                )
        except asyncio.TimeoutError:
            logger.warning("Literature search timed out", query=query[:120])
            return []
        except Exception as e:
            logger.warning("Literature match failed", error=str(e))
            return []

    def _format_literature_section(self, matches: List) -> str:
        lines = [
            "The following published studies are comparable to this analysis "
            "(automated literature matching via PubMed index):"
        ]
        for i, m in enumerate(matches, 1):
            authors = m.authors[0] if m.authors else "Unknown"
            et_al = f"{authors} et al." if len(m.authors) > 1 else authors
            year = m.year or "n.d."
            lines.append(
                f"{i}. {et_al} ({year}). {m.title}. "
                f"{m.journal or 'Journal unknown'}. "
                f"PMID: {m.pmid or 'N/A'}. "
                f"Relevance: {m.similarity_score:.2f}."
            )
        return "\n".join(lines)
