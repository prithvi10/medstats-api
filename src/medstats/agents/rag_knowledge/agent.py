"""RAGKnowledgeAgent — utility agent for grounded knowledge retrieval."""
from __future__ import annotations

from typing import List, Optional

from src.medstats.rag.corpus.ingest import is_rag_available
from src.medstats.rag.models import RAGResult
from src.medstats.rag.retriever import ParentChildRetriever
from src.shared.config.settings import get_settings
from src.shared.db.session import async_session_factory
from src.shared.logging.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


class RAGKnowledgeAgent:
    """
    Utility agent invoked by other agents to retrieve grounded statistical knowledge.
    Not a pipeline step — call retrieve() from within other agents.
    """

    role_name = "RAGKnowledge"

    async def retrieve(
        self,
        query: str,
        tags: Optional[List[str]] = None,
        top_k: Optional[int] = None,
    ) -> RAGResult:
        if not is_rag_available() or not settings.OPENAI_API_KEY:
            logger.debug("RAG unavailable — returning empty result")
            return RAGResult(passages=[], citations=[])

        try:
            async with async_session_factory() as session:
                retriever = ParentChildRetriever(session)
                return await retriever.retrieve(query=query, tags=tags, top_k=top_k)
        except Exception as e:
            logger.warning("RAG retrieval failed", error=str(e), query=query[:80])
            return RAGResult(passages=[], citations=[])

    def format_context_block(self, result: RAGResult) -> str:
        """Format retrieved passages for injection into LLM prompts."""
        if not result.passages:
            return ""
        lines = ["RETRIEVED KNOWLEDGE (grounded, with citations):"]
        for i, passage in enumerate(result.passages, 1):
            lines.append(f"[{i}] {passage}")
        return "\n".join(lines)


def get_rag_agent() -> Optional[RAGKnowledgeAgent]:
    if not settings.RAG_ENABLED:
        return None
    return RAGKnowledgeAgent()
