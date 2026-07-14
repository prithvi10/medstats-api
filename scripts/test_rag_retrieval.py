#!/usr/bin/env python3
"""
Evaluate RAG retrieval against the live corpus for sample masterchart/thesis intents.

Usage:
  python scripts/test_rag_retrieval.py
  python scripts/test_rag_retrieval.py --case 05_two_group_ttest
  python scripts/test_rag_retrieval.py --json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.rag_eval.cases import CASE_BY_ID, RAG_EVAL_CASES
from tests.rag_eval.runner import corpus_stats, evaluate_case, rag_prerequisites_met, run_all_cases


def _print_case(result: dict) -> None:
    k = result["knowledge"]
    lit = result["literature"]
    status = "PASS" if result["passed"] else "FAIL"
    print(f"\n{'=' * 72}")
    print(f"[{status}] {result['case_id']} — {result['title']}")
    print(f"  Design: {result['study_design']}  Tests: {', '.join(result['test_types'])}")

    print("\n  KNOWLEDGE (textbook / guidelines)")
    print(f"  Query: {k['query'][:120]}...")
    print(f"  Passages: {k['passage_count']}  Keyword match: {k['keyword_hit']}  Source hint: {k['source_hit']}")
    for i, passage in enumerate(k["passages"][:2], 1):
        snippet = passage.replace("\n", " ")[:220]
        print(f"    [{i}] {snippet}...")
    for cite in k["citations"][:2]:
        page = f", p.{cite['page']}" if cite.get("page") else ""
        print(f"    cite: {cite['source']}{page}")

    print("\n  LITERATURE (PubMed index)")
    print(f"  Query: {lit['query'][:120]}...")
    print(f"  Disciplines: {lit['disciplines']}  Methods tags: {lit['methods_tags']}")
    print(f"  Matches: {lit['match_count']}")
    for m in lit["matches"][:3]:
        print(
            f"    • ({m['similarity_score']}) {m['title'][:70]}… "
            f"[{m.get('year')}] PMID {m.get('pmid')}"
        )
    if lit.get("section_preview"):
        first_line = lit["section_preview"].splitlines()[1] if "\n" in lit["section_preview"] else lit["section_preview"]
        print(f"  Report snippet: {first_line[:160]}...")


async def main() -> int:
    parser = argparse.ArgumentParser(description="Test RAG retrieval for sample thesis scenarios")
    parser.add_argument("--case", help="Run a single case id (e.g. 05_two_group_ttest)")
    parser.add_argument("--json", action="store_true", help="Print JSON summary")
    args = parser.parse_args()

    if not rag_prerequisites_met():
        print(
            "RAG prerequisites not met.\n"
            "  • DATABASE_URL must be PostgreSQL with pgvector\n"
            "  • OPENAI_API_KEY must be set\n"
            "  • Run: python scripts/build_rag_corpus.py --seed-only --bootstrap-text\n"
            "         python scripts/build_rag_corpus.py --pdf corpus/pdfs/openintro-biostatistics.pdf ...\n"
            "         python scripts/build_rag_corpus.py --pubmed --pubmed-max 200",
            file=sys.stderr,
        )
        return 2

    stats = await corpus_stats()
    if stats["rag_documents"] < 10 or stats["literature_index"] < 10:
        print(f"Corpus too small: {stats}. Build the RAG DB first.", file=sys.stderr)
        return 2

    if args.case:
        case = CASE_BY_ID.get(args.case)
        if not case:
            print(f"Unknown case: {args.case}", file=sys.stderr)
            return 1
        payload = {"corpus": stats, "results": [await evaluate_case(case)]}
        payload["passed"] = 1 if payload["results"][0]["passed"] else 0
        payload["failed"] = 1 - payload["passed"]
        payload["total"] = 1
    else:
        payload = await run_all_cases()

    if args.json:
        print(json.dumps(payload, indent=2, default=str))
    else:
        print("RAG corpus:", stats)
        print(f"Cases: {payload['passed']}/{payload['total']} passed")
        for result in payload["results"]:
            _print_case(result)

    return 0 if payload["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
