"""Run RAG knowledge + literature retrieval checks against live Postgres corpus."""
from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from sqlalchemy import text

from src.medstats.agents.literature_match.agent import LiteratureMatchAgent
from src.medstats.agents.rag_knowledge.agent import RAGKnowledgeAgent
from src.medstats.rag.corpus.ingest import is_rag_available
from src.shared.config.settings import get_settings
from src.shared.db.session import async_session_factory

from tests.rag_eval.cases import RAGEvalCase

settings = get_settings()


def rag_prerequisites_met() -> bool:
    if not is_rag_available():
        return False
    if not settings.OPENAI_API_KEY:
        return False
    if "sqlite" in settings.DATABASE_URL:
        return False
    return True


async def corpus_stats() -> Dict[str, int]:
    async with async_session_factory() as session:
        rag = (await session.execute(text("SELECT COUNT(*) FROM rag_documents"))).scalar_one()
        lit = (await session.execute(text("SELECT COUNT(*) FROM literature_index"))).scalar_one()
        nodes = (await session.execute(text("SELECT COUNT(*) FROM knowledge_nodes"))).scalar_one()
    return {
        "rag_documents": int(rag),
        "literature_index": int(lit),
        "knowledge_nodes": int(nodes),
    }


def _knowledge_query(case: RAGEvalCase) -> str:
    if case.knowledge_query:
        return case.knowledge_query
    test_label = case.test_types[0] if case.test_types else "statistical test"
    return (
        f"when and why to use {test_label} in a {case.study_design} medical study "
        f"with N={case.sample_size}. {case.intent[:200]}"
    )


def _literature_agent_data(case: RAGEvalCase) -> dict:
    test_results = []
    for var in case.variable_names:
        test_type = case.test_types[0] if len(case.test_types) == 1 else None
        if len(case.test_types) > 1 and var in case.significant_variables:
            test_type = case.test_types[0] if "PAIN" in var else case.test_types[-1]
        test_results.append(
            {
                "variable_name": var,
                "test_result": {
                    "test_type": test_type or case.test_types[0],
                    "is_significant_05": var in case.significant_variables,
                },
            }
        )
    return {
        "title": case.title,
        "intent": case.intent,
        "study_design": case.study_design,
        "sample_size": case.sample_size,
        "test_results": test_results,
        "variables": [{"name": v} for v in case.variable_names],
    }


def _text_contains_any(text: str, needles: List[str]) -> bool:
    lower = text.lower()
    return any(n.lower() in lower for n in needles)


async def evaluate_knowledge(case: RAGEvalCase, top_k: int = 3) -> Dict[str, Any]:
    agent = RAGKnowledgeAgent()
    query = _knowledge_query(case)
    result = await agent.retrieve(query, tags=case.knowledge_tags or None, top_k=top_k)

    combined = " ".join(result.passages).lower()
    keyword_hit = _text_contains_any(combined, case.knowledge_must_contain) if case.knowledge_must_contain else True
    source_hit = True
    if case.knowledge_source_hints and result.citations:
        cites = " ".join(c.source for c in result.citations).lower()
        source_hit = _text_contains_any(cites, case.knowledge_source_hints)
    elif case.knowledge_source_hints and result.passages:
        source_hit = _text_contains_any(combined, case.knowledge_source_hints)

    passed = bool(result.passages) and keyword_hit

    return {
        "case_id": case.id,
        "query": query,
        "passed": passed,
        "passage_count": len(result.passages),
        "keyword_hit": keyword_hit,
        "source_hit": source_hit,
        "passages": result.passages,
        "citations": [{"source": c.source, "page": c.page, "section": c.section} for c in result.citations],
        "context_block": agent.format_context_block(result),
    }


async def evaluate_literature(case: RAGEvalCase) -> Dict[str, Any]:
    agent = LiteratureMatchAgent(services=None)
    data = _literature_agent_data(case)
    query = agent._build_query(data)
    disciplines = agent._infer_disciplines(data)
    methods_tags = agent._extract_methods_tags(data)
    matches = await agent._search_literature(query, disciplines, methods_tags)

    scored = [m for m in matches if m.similarity_score >= case.literature_min_score]
    passed = len(scored) >= case.literature_min_matches

    return {
        "case_id": case.id,
        "query": query,
        "disciplines": disciplines,
        "methods_tags": methods_tags,
        "passed": passed,
        "match_count": len(matches),
        "matches": [
            {
                "pmid": m.pmid,
                "title": m.title,
                "year": m.year,
                "journal": m.journal,
                "similarity_score": round(m.similarity_score, 3),
                "abstract_snippet": m.abstract_snippet[:180],
            }
            for m in matches
        ],
        "section_preview": agent._format_literature_section(matches) if matches else "",
    }


async def evaluate_case(case: RAGEvalCase) -> Dict[str, Any]:
    knowledge, literature = await asyncio.gather(
        evaluate_knowledge(case),
        evaluate_literature(case),
    )
    return {
        "case_id": case.id,
        "title": case.title,
        "study_design": case.study_design,
        "test_types": case.test_types,
        "knowledge": knowledge,
        "literature": literature,
        "passed": knowledge["passed"] and literature["passed"],
    }


async def run_all_cases(cases: Optional[List[RAGEvalCase]] = None) -> Dict[str, Any]:
    from tests.rag_eval.cases import RAG_EVAL_CASES

    cases = cases or RAG_EVAL_CASES
    stats = await corpus_stats()
    results = []
    for case in cases:
        results.append(await evaluate_case(case))

    passed = sum(1 for r in results if r["passed"])
    return {
        "corpus": stats,
        "total": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "results": results,
    }
