"""Integration-style tests for RAGKnowledgeAgent (mocked retrieval)."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch

from src.medstats.agents.rag_knowledge.agent import RAGKnowledgeAgent
from src.medstats.rag.models import RAGResult, Citation


@pytest.mark.asyncio
async def test_rag_agent_format_context_block():
    agent = RAGKnowledgeAgent()
    result = RAGResult(
        passages=["Fisher's exact is preferred when expected cells < 5. [ICMR 2022]"],
        citations=[Citation(source="ICMR Statistical Guidelines 2022", page=47)],
    )
    block = agent.format_context_block(result)
    assert "RETRIEVED KNOWLEDGE" in block
    assert "Fisher" in block


@pytest.mark.asyncio
async def test_rag_agent_retrieve_returns_empty_when_disabled():
    agent = RAGKnowledgeAgent()
    with patch("src.medstats.agents.rag_knowledge.agent.is_rag_available", return_value=False):
        result = await agent.retrieve("chi-square test assumptions")
    assert result.passages == []


@pytest.mark.asyncio
async def test_literature_match_builds_query():
    from src.medstats.agents.literature_match.agent import LiteratureMatchAgent

    agent = LiteratureMatchAgent(services=None)
    query = agent._build_query({
        "title": "Fracture healing study",
        "intent": "Compare recovery times",
        "study_design": "two_group",
        "sample_size": 60,
        "test_results": [
            {"variable_name": "RECOVERY", "test_result": {"is_significant_05": True}},
        ],
    })
    assert "Fracture" in query
    assert "RECOVERY" in query
