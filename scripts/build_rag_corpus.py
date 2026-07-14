#!/usr/bin/env python3
"""
One-time bootstrap: seed knowledge graph + ingest PDF/text corpus.

Usage:
  python scripts/build_rag_corpus.py --seed-only
  python scripts/build_rag_corpus.py --pdf path/to/icmr.pdf --source-type icmr_guideline
  python scripts/build_rag_corpus.py --bootstrap-text   # minimal text corpus for dev/tests
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.shared.db.session import async_session_factory
from src.medstats.rag.corpus.ingest import ingest_pdf, ingest_text_corpus, seed_knowledge_graph
from src.medstats.rag.corpus.pubmed import PubMedCorpusIndexer
from src.medstats.rag.embedder import ChunkEmbedder
from src.shared.config.settings import get_settings

settings = get_settings()

BOOTSTRAP_TEXT = [
    {
        "source_name": "ICMR Statistical Guidelines 2022 (excerpt)",
        "source_type": "icmr_guideline",
        "text": """
        Statistical Analysis Guidelines for Biomedical Research.

        Normality Testing: For sample sizes less than 30, the Shapiro-Wilk test should be
        performed before applying parametric tests such as the unpaired t-test or one-way ANOVA.
        If normality is violated (p < 0.05), use Mann-Whitney U test for two groups or
        Kruskal-Wallis test for three or more groups.

        Categorical Data: Chi-square test is appropriate when expected cell frequencies are
        at least 5 in all cells. When any expected cell count is below 5, Fisher's exact test
        is preferred over chi-square as the chi-square approximation is unreliable.

        Sample Size: Studies with N < 20 have limited statistical power. Non-parametric
        alternatives should be considered and limitations noted in the discussion.
        """,
    },
    {
        "source_name": "Biostatistics Reference (excerpt)",
        "source_type": "textbook",
        "text": """
        Mann-Whitney U Test: The non-parametric equivalent of the independent samples t-test.
        Used to compare two independent groups when data are ordinal or continuous but not
        normally distributed. Does not assume normality. Also called Wilcoxon rank-sum test.

        Fisher's Exact Test: Exact test for association in 2x2 or larger contingency tables
        when sample sizes are small. Preferred when chi-square expected frequencies are below 5.

        Bonferroni Correction: When performing multiple comparisons, adjust alpha by dividing
        by the number of tests to control family-wise Type I error rate.
        """,
    },
]


async def main() -> None:
    parser = argparse.ArgumentParser(description="Build GraphRAG corpus")
    parser.add_argument("--seed-only", action="store_true", help="Only seed knowledge graph")
    parser.add_argument("--bootstrap-text", action="store_true", help="Ingest minimal text corpus")
    parser.add_argument("--pdf", type=str, help="Path to PDF to ingest")
    parser.add_argument("--source-type", type=str, default="textbook")
    parser.add_argument("--pubmed", action="store_true", help="Index PubMed abstracts (all disciplines)")
    parser.add_argument("--pubmed-max", type=int, default=100, help="Max abstracts per discipline")
    parser.add_argument("--source-name", type=str, default=None)
    args = parser.parse_args()

    embedder = ChunkEmbedder() if settings.OPENAI_API_KEY else None

    async with async_session_factory() as session:
        n = await seed_knowledge_graph(session, embedder=embedder)
        print(f"Knowledge graph seeded: {n} items")

        if args.bootstrap_text:
            count = await ingest_text_corpus(session, BOOTSTRAP_TEXT, embedder=embedder)
            print(f"Bootstrap text corpus: {count} chunks")

        if args.pdf:
            count = await ingest_pdf(
                session,
                args.pdf,
                source_type=args.source_type,
                source_name=args.source_name,
                embedder=embedder,
            )
            print(f"PDF ingested: {count} chunks")

        if args.pubmed:
            indexer = PubMedCorpusIndexer(session, embedder=embedder)
            count = await indexer.index_all_disciplines(max_per_discipline=args.pubmed_max)
            print(f"PubMed indexed: {count} abstracts")

        if args.seed_only and not args.bootstrap_text and not args.pdf and not args.pubmed:
            print("Done (--seed-only)")


if __name__ == "__main__":
    asyncio.run(main())
