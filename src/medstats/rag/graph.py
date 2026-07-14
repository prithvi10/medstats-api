"""Knowledge graph traversal for GraphRAG expansion."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from sqlalchemy import or_, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from src.medstats.db.rag_models import KnowledgeEdge, KnowledgeNode, RAGDocument
from src.shared.logging.logger import get_logger

logger = get_logger(__name__)


class GraphExpander:
    """Expand retrieval via 1-2 hop knowledge graph traversal."""

    def __init__(self, session: AsyncSession, max_hops: int = 2):
        self.session = session
        self.max_hops = max_hops

    async def expand_from_tags(self, tags: List[str], limit: int = 15) -> List[Dict[str, Any]]:
        """Find related concepts from tags and return associated document snippets."""
        if not tags:
            return []

        node_names = await self._resolve_nodes_from_tags(tags)
        if not node_names:
            return []

        expanded_tags = await self._traverse_graph(node_names)
        return await self._fetch_docs_by_tags(list(expanded_tags), limit=limit)

    async def _resolve_nodes_from_tags(self, tags: List[str]) -> Set[str]:
        """Map tag strings to knowledge node names."""
        names: Set[str] = set()
        tag_to_name = {
            "chi_square": "Chi-square test",
            "fisher_exact": "Fisher's exact test",
            "mann_whitney_u": "Mann-Whitney U test",
            "unpaired_t_test": "Student's unpaired t-test",
            "paired_t_test": "Student's paired t-test",
            "one_way_anova": "One-way ANOVA",
            "kruskal_wallis": "Kruskal-Wallis test",
            "shapiro_wilk": "Shapiro-Wilk test",
            "z_test_proportion": "Z-test of proportions",
            "normality": "Normality assumption",
            "single_cohort": "Single cohort study",
            "two_group": "Two-group comparison",
            "multi_group": "Multi-group comparison",
        }
        for tag in tags:
            if tag in tag_to_name:
                names.add(tag_to_name[tag])
        return names

    async def _traverse_graph(self, start_names: Set[str]) -> Set[str]:
        """BFS traversal up to max_hops from starting node names."""
        result_tags: Set[str] = set(start_names)
        current_ids: Set[str] = set()

        for name in start_names:
            row = await self.session.execute(
                select(KnowledgeNode.id).where(KnowledgeNode.name == name)
            )
            nid = row.scalar_one_or_none()
            if nid:
                current_ids.add(str(nid))

        visited: Set[str] = set(current_ids)
        for _ in range(self.max_hops):
            if not current_ids:
                break
            next_ids: Set[str] = set()
            for node_id in current_ids:
                edges = await self.session.execute(
                    select(KnowledgeEdge.to_node, KnowledgeEdge.from_node).where(
                        or_(
                            KnowledgeEdge.from_node == node_id,
                            KnowledgeEdge.to_node == node_id,
                        )
                    )
                )
                for to_node, from_node in edges.all():
                    neighbor = str(to_node) if str(from_node) == node_id else str(from_node)
                    if neighbor not in visited:
                        visited.add(neighbor)
                        next_ids.add(neighbor)

            if next_ids:
                nodes = await self.session.execute(
                    select(KnowledgeNode.name).where(
                        KnowledgeNode.id.in_(list(next_ids))
                    )
                )
                for (name,) in nodes.all():
                    result_tags.add(name)
            current_ids = next_ids

        return result_tags

    async def _fetch_docs_by_tags(self, tags: List[str], limit: int) -> List[Dict[str, Any]]:
        """Fetch rag_documents whose metadata contains any of the tags."""
        if not tags:
            return []

        # JSONB overlap search on concept_tags / test_tags
        conditions = []
        for tag in tags[:10]:
            tag_clean = tag.lower().replace(" ", "_").replace("-", "_")
            conditions.append(
                text(
                    "doc_metadata->'concept_tags' ? :tag OR doc_metadata->'test_tags' ? :tag"
                ).bindparams(tag=tag_clean)
            )

        query = select(RAGDocument).where(or_(*conditions)).limit(limit)
        try:
            rows = await self.session.execute(query)
        except Exception as e:
            logger.warning("Graph doc fetch failed", error=str(e))
            return []

        results = []
        for doc in rows.scalars().all():
            results.append({
                "doc_id": str(doc.id),
                "content": doc.content,
                "parent_content": doc.parent_content or doc.content,
                "source_name": doc.source_name,
                "source_type": doc.source_type or "document",
                "score": 0.3,
                "doc_metadata": doc.doc_metadata,
            })
        return results
