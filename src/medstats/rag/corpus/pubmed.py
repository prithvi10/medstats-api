"""
PubMed corpus indexer — all medical science disciplines.

Uses NCBI Entrez API (free). Indexes abstracts with discipline and methods tags.
"""
from __future__ import annotations

import asyncio
import re
import time
import uuid
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.medstats.db.rag_models import LiteratureIndex, serialize_embedding
from src.medstats.rag.chunking import sanitize_text
from src.medstats.rag.chunking_utils import TEST_TAGS, extract_tags
from src.medstats.rag.embedder import ChunkEmbedder
from src.medstats.rag.corpus.ncbi_entrez import configure_entrez
from src.shared.config.settings import get_settings
from src.shared.logging.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

# Comprehensive discipline coverage for all medical sciences
DISCIPLINE_QUERIES: Dict[str, str] = {
    "general_medicine": '("general medicine"[MeSH] OR "internal medicine"[MeSH]) AND (statistical OR biostatistic*)',
    "orthopaedics": '"orthopedics"[MeSH] AND (statistical OR "sample size")',
    "dentistry": '"dentistry"[MeSH] AND (statistical OR biostatistic*)',
    "physiotherapy": '"physical therapy modalities"[MeSH] OR "physical therapy"[MeSH] AND statistical',
    "ophthalmology": '"ophthalmology"[MeSH] AND statistical',
    "obstetrics_gynaecology": '("obstetrics"[MeSH] OR "gynecology"[MeSH]) AND statistical',
    "paediatrics": '"pediatrics"[MeSH] AND statistical',
    "ent_surgery": '("otolaryngology"[MeSH] OR "general surgery"[MeSH]) AND statistical',
    "psychiatry": '"psychiatry"[MeSH] AND statistical',
    "radiology_pathology": '("radiology"[MeSH] OR "pathology"[MeSH]) AND statistical',
    "nursing": '"nursing"[MeSH] AND statistical',
    "pharmacy": '"pharmacy"[MeSH] AND statistical',
    "ayurveda": '("medicine, ayurvedic"[MeSH] OR ayurveda) AND statistical',
    "community_medicine": '"public health"[MeSH] AND statistical',
}


def extract_methods_tags(text: str) -> List[str]:
    """Extract statistical method tags from abstract text."""
    tags = extract_tags(text)
    methods = [t for t in tags if t in TEST_TAGS]
    lower = text.lower()
    if re.search(r"\bn\s*[=<]\s*\d+", lower) or "sample size" in lower:
        n_match = re.search(r"(?:n\s*=\s*|sample size of\s*)(\d+)", lower)
        if n_match:
            n = int(n_match.group(1))
            if n < 30:
                methods.append("small_sample")
            if n < 100:
                methods.append("n_lt_100")
    if "randomized" in lower or "rct" in lower:
        methods.append("rct")
    if "retrospective" in lower:
        methods.append("retrospective")
    if "prospective" in lower:
        methods.append("prospective")
    return list(set(methods))


class PubMedCorpusIndexer:
    """Fetch and index PubMed abstracts into literature_index."""

    def __init__(
        self,
        session: AsyncSession,
        email: Optional[str] = None,
        embedder: Optional[ChunkEmbedder] = None,
    ):
        self.session = session
        self.email = email or settings.NCBI_EMAIL
        self.embedder = embedder or ChunkEmbedder()
        self._last_request = 0.0
        configure_entrez(self.email)

    async def index_all_disciplines(
        self,
        max_per_discipline: int = 500,
        batch_size: int = 100,
    ) -> int:
        """Index abstracts for all configured medical disciplines."""
        total = 0
        for discipline, query in DISCIPLINE_QUERIES.items():
            count = await self.index_discipline(
                discipline=discipline,
                query=query,
                max_results=max_per_discipline,
                batch_size=batch_size,
            )
            total += count
            logger.info("Discipline indexed", discipline=discipline, count=count)
        return total

    async def index_discipline(
        self,
        discipline: str,
        query: str,
        max_results: int = 500,
        batch_size: int = 100,
    ) -> int:
        pmids = await asyncio.to_thread(
            self._search_pmids, query, max_results
        )
        if not pmids:
            return 0

        indexed = 0
        for i in range(0, len(pmids), batch_size):
            batch_pmids = pmids[i : i + batch_size]
            records = await asyncio.to_thread(self._fetch_records, batch_pmids)
            count = await self._store_records(records, discipline)
            indexed += count

        await self.session.commit()
        return indexed

    def _rate_limit(self) -> None:
        elapsed = time.time() - self._last_request
        if elapsed < 0.34:
            time.sleep(0.34 - elapsed)
        self._last_request = time.time()

    def _search_pmids(self, query: str, max_results: int) -> List[str]:
        from Bio import Entrez

        configure_entrez(self.email)
        self._rate_limit()
        handle = Entrez.esearch(
            db="pubmed",
            term=query,
            retmax=max_results,
            sort="relevance",
        )
        record = Entrez.read(handle)
        handle.close()
        return record.get("IdList", [])

    def _fetch_records(self, pmids: List[str]) -> List[Dict[str, Any]]:
        from Bio import Entrez

        configure_entrez(self.email)
        self._rate_limit()
        handle = Entrez.efetch(db="pubmed", id=",".join(pmids), rettype="xml", retmode="xml")
        records = Entrez.read(handle)
        handle.close()

        articles = []
        for article in records.get("PubmedArticle", []):
            medline = article.get("MedlineCitation", {})
            pubmed_article = medline.get("Article", {})
            pmid = str(medline.get("PMID", ""))
            title = pubmed_article.get("ArticleTitle", "")
            abstract_parts = pubmed_article.get("Abstract", {}).get("AbstractText", [])
            if isinstance(abstract_parts, list):
                abstract = " ".join(str(p) for p in abstract_parts)
            else:
                abstract = str(abstract_parts) if abstract_parts else ""

            authors = []
            for author in pubmed_article.get("AuthorList", []):
                last = author.get("LastName", "")
                fore = author.get("ForeName", "")
                if last:
                    authors.append(f"{fore} {last}".strip())

            journal = pubmed_article.get("Journal", {}).get("Title", "")
            year = None
            pub_date = pubmed_article.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})
            if pub_date.get("Year"):
                try:
                    year = int(pub_date["Year"])
                except (ValueError, TypeError):
                    pass

            mesh = []
            for desc in medline.get("MeshHeadingList", []):
                descriptor = desc.get("DescriptorName", "")
                if descriptor:
                    mesh.append(str(descriptor))

            articles.append({
                "pmid": pmid,
                "title": sanitize_text(str(title)),
                "abstract": sanitize_text(abstract),
                "authors": authors,
                "year": year,
                "journal": str(journal),
                "mesh_terms": mesh,
            })
        return articles

    async def _store_records(self, records: List[Dict[str, Any]], discipline: str) -> int:
        if not records:
            return 0

        texts_for_embed = []
        to_insert = []

        for rec in records:
            existing = await self.session.execute(
                select(LiteratureIndex.id).where(LiteratureIndex.pmid == rec["pmid"])
            )
            if existing.scalar_one_or_none():
                continue

            abstract = sanitize_text(rec.get("abstract") or "")
            title = sanitize_text(rec["title"])
            embed_text = f"{title}. {abstract}"
            methods_tags = extract_methods_tags(embed_text)

            to_insert.append({
                "rec": rec,
                "methods_tags": methods_tags,
                "embed_text": embed_text,
            })
            texts_for_embed.append(embed_text)

        if not to_insert:
            return 0

        embeddings: List[List[float]] = []
        if settings.OPENAI_API_KEY:
            try:
                embeddings = await self.embedder.embed_texts(texts_for_embed)
            except Exception as e:
                logger.warning("Literature embedding failed", error=str(e))
                embeddings = [[] for _ in to_insert]
        else:
            embeddings = [[] for _ in to_insert]

        for i, item in enumerate(to_insert):
            rec = item["rec"]
            self.session.add(
                LiteratureIndex(
                    id=uuid.uuid4(),
                    pmid=rec["pmid"],
                    title=rec["title"],
                    abstract=rec.get("abstract"),
                    authors=rec.get("authors"),
                    year=rec.get("year"),
                    journal=rec.get("journal"),
                    mesh_terms=rec.get("mesh_terms"),
                    methods_tags=item["methods_tags"],
                    disciplines=[discipline],
                    embedding=serialize_embedding(
                        embeddings[i] if i < len(embeddings) and embeddings[i] else None
                    ),
                )
            )

        return len(to_insert)
