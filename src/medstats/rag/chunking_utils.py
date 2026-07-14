"""Tag extraction and structural section detection for academic documents."""
from __future__ import annotations

import re
from typing import List, Set

# Statistical tests (aligned with medstats TEST_NAME_MAP keys)
TEST_TAGS = {
    "z_test_proportion": ["z-test", "z test", "proportion", "z_test_proportion"],
    "chi_square": ["chi-square", "chi square", "chi_square", "χ2", "chi-squared"],
    "fisher_exact": ["fisher", "fisher's exact", "fisher_exact", "fisher exact"],
    "unpaired_t_test": ["unpaired t-test", "student t-test", "independent t-test", "unpaired_t_test"],
    "paired_t_test": ["paired t-test", "paired_t_test", "dependent t-test"],
    "mann_whitney_u": ["mann-whitney", "mann whitney", "wilcoxon rank-sum", "mann_whitney_u"],
    "one_way_anova": ["anova", "one-way anova", "one_way_anova", "analysis of variance"],
    "kruskal_wallis": ["kruskal-wallis", "kruskal wallis", "kruskal_wallis"],
    "shapiro_wilk": ["shapiro-wilk", "shapiro wilk", "normality test", "shapiro_wilk"],
}

STUDY_DESIGN_TAGS = {
    "single_cohort": ["single cohort", "single group", "descriptive study", "cross-sectional"],
    "two_group": ["two group", "two-group", "comparative", "case-control", "randomized"],
    "multi_group": ["multi group", "multi-group", "three groups", "multiple groups"],
}

CONCEPT_TAGS = [
    "normality", "p-value", "p value", "significance", "confidence interval",
    "sample size", "power", "bonferroni", "multiple testing", "assumption",
    "parametric", "non-parametric", "nonparametric", "effect size",
    "type i error", "type ii error", "degrees of freedom",
]

STRUCTURAL_HEADERS = {
    "abstract": ["abstract", "summary"],
    "methodology": ["methodology", "methods", "materials and methods", "statistical analysis"],
    "results": ["results", "findings", "observations"],
    "discussion": ["discussion", "interpretation"],
    "introduction": ["introduction", "background"],
    "definition": ["definition", "theorem", "formula", "equation"],
}


def extract_tags(text: str) -> List[str]:
    """Extract statistical test, design, and concept tags from text."""
    lower = text.lower()
    found: Set[str] = set()

    for tag, keywords in TEST_TAGS.items():
        if any(kw in lower for kw in keywords):
            found.add(tag)

    for tag, keywords in STUDY_DESIGN_TAGS.items():
        if any(kw in lower for kw in keywords):
            found.add(tag)

    for concept in CONCEPT_TAGS:
        if concept in lower:
            found.add(concept.replace(" ", "_").replace("-", "_"))

    return sorted(found)


def detect_structural_type(heading: str) -> str:
    """Map a heading to a structural section type."""
    lower = heading.lower().strip()
    for stype, keywords in STRUCTURAL_HEADERS.items():
        if any(kw in lower for kw in keywords):
            return stype
    return "body"


def _hard_split(text: str, max_chars: int, overlap: int) -> List[str]:
    """Split oversized text when paragraph boundaries are missing (common in PDFs)."""
    if len(text) <= max_chars:
        return [text]

    parts: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            # Prefer breaking at whitespace near the limit
            break_at = text.rfind(" ", start + max_chars // 2, end)
            if break_at > start:
                end = break_at
        parts.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)

    return [p for p in parts if p]


def _paragraphs_from_text(text: str, max_chars: int, overlap: int) -> List[str]:
    """Extract paragraphs; PDF line breaks often lack blank lines between paragraphs."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        paragraphs = [p.strip() for p in text.splitlines() if p.strip()]
    if not paragraphs:
        return _hard_split(text.strip(), max_chars, overlap)

    normalized: List[str] = []
    for para in paragraphs:
        if len(para) <= max_chars:
            normalized.append(para)
        else:
            normalized.extend(_hard_split(para, max_chars, overlap))
    return normalized


def split_into_child_chunks(text: str, max_chars: int = 1200, overlap: int = 100) -> List[str]:
    """Split text into child chunks at paragraph boundaries."""
    if not text or not text.strip():
        return []

    paragraphs = _paragraphs_from_text(text.strip(), max_chars, overlap)

    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para)
        if para_len > max_chars:
            if current:
                chunks.append("\n\n".join(current))
                current = []
                current_len = 0
            chunks.extend(_hard_split(para, max_chars, overlap))
            continue

        if current_len + para_len + 2 > max_chars and current:
            chunks.append("\n\n".join(current))
            # overlap: keep last paragraph for context
            if overlap and current:
                current = [current[-1], para] if len(current[-1]) < overlap else [para]
                current_len = sum(len(p) for p in current) + 2 * (len(current) - 1)
            else:
                current = [para]
                current_len = para_len
        else:
            current.append(para)
            current_len += para_len + 2

    if current:
        chunk = "\n\n".join(current)
        if len(chunk) > max_chars:
            chunks.extend(_hard_split(chunk, max_chars, overlap))
        else:
            chunks.append(chunk)

    return chunks
