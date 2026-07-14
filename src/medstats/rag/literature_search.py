"""Literature index search for automated literature matching."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.medstats.rag.models import LiteratureMatch
from src.shared.config.settings import get_settings
from src.shared.logging.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


class LiteratureSearchEngine:
    """Semantic search over literature_index (PubMed abstracts)."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def search(
        self,
        query_embedding: List[float],
        disciplines: Optional[List[str]] = None,
        methods_tags: Optional[List[str]] = None,
        limit: int = 5,
    ) -> List[LiteratureMatch]:
        if not query_embedding or "sqlite" in settings.DATABASE_URL:
            return await self._keyword_fallback(disciplines, methods_tags, limit)

        try:
            emb_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
            params: Dict[str, Any] = {"emb": emb_str, "limit": limit}
            filters = ["embedding IS NOT NULL"]

            if disciplines:
                filters.append("disciplines && :disciplines")
                params["disciplines"] = disciplines
            if methods_tags:
                filters.append("methods_tags && :methods_tags")
                params["methods_tags"] = methods_tags

            where = " AND ".join(filters)
            sql = text(f"""
                SELECT pmid, title, authors, year, journal, abstract,
                       1 - (embedding <=> CAST(:emb AS vector)) AS score
                FROM literature_index
                WHERE {where}
                ORDER BY embedding <=> CAST(:emb AS vector)
                LIMIT :limit
            """)
            rows = (await self.session.execute(sql, params)).mappings()
        except Exception as e:
            logger.warning("Literature search failed", error=str(e))
            return []

        return [
            LiteratureMatch(
                pmid=r["pmid"],
                title=r["title"],
                authors=list(r["authors"] or []),
                year=r["year"],
                journal=r["journal"],
                abstract_snippet=(r["abstract"] or "")[:300],
                similarity_score=float(r["score"]),
            )
            for r in rows
        ]

    async def _keyword_fallback(
        self,
        disciplines: Optional[List[str]],
        methods_tags: Optional[List[str]],
        limit: int,
    ) -> List[LiteratureMatch]:
        return []
