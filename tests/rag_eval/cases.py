"""Scenario-based RAG evaluation cases (masterchart / thesis intents)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from tests.scenarios.definitions import SCENARIO_BY_ID, StudentScenario


@dataclass
class RAGEvalCase:
    """One retrieval probe mirroring a student submission context."""

    id: str
    title: str
    intent: str
    study_design: str
    sample_size: int
    test_types: List[str]
    variable_names: List[str]
    significant_variables: List[str] = field(default_factory=list)
    knowledge_query: Optional[str] = None
    knowledge_tags: List[str] = field(default_factory=list)
    knowledge_must_contain: List[str] = field(default_factory=list)
    knowledge_source_hints: List[str] = field(default_factory=list)
    literature_min_matches: int = 1
    literature_min_score: float = 0.25


def _from_scenario(
    scenario: StudentScenario,
    *,
    sample_size: int,
    test_types: List[str],
    variable_names: List[str],
    significant_variables: Optional[List[str]] = None,
    knowledge_must_contain: List[str],
    knowledge_tags: Optional[List[str]] = None,
    knowledge_source_hints: Optional[List[str]] = None,
) -> RAGEvalCase:
    sig = significant_variables or variable_names[:1]
    tags = knowledge_tags or test_types + [scenario.study_design]
    return RAGEvalCase(
        id=scenario.id,
        title=scenario.title,
        intent=scenario.intent,
        study_design=scenario.study_design,
        sample_size=sample_size,
        test_types=test_types,
        variable_names=variable_names,
        significant_variables=sig,
        knowledge_tags=tags,
        knowledge_must_contain=knowledge_must_contain,
        knowledge_source_hints=knowledge_source_hints or ["OpenIntro", "Biostatistics", "ICMR"],
    )


RAG_EVAL_CASES: List[RAGEvalCase] = [
    _from_scenario(
        SCENARIO_BY_ID["05_two_group_ttest"],
        sample_size=60,
        test_types=["unpaired_t_test"],
        variable_names=["RECOVERY DAYS"],
        knowledge_must_contain=["t-test", "t test", "independent", "two group", "two-group"],
        knowledge_tags=["unpaired_t_test", "two_group"],
    ),
    _from_scenario(
        SCENARIO_BY_ID["06_two_group_mann_whitney"],
        sample_size=48,
        test_types=["mann_whitney_u"],
        variable_names=["HOSPITAL STAY"],
        knowledge_must_contain=["mann-whitney", "mann whitney", "non-parametric", "nonparametric", "wilcoxon", "rank", "comparison", "groups"],
        knowledge_tags=["mann_whitney_u", "two_group"],
    ),
    _from_scenario(
        SCENARIO_BY_ID["07_two_group_chi_square"],
        sample_size=80,
        test_types=["chi_square"],
        variable_names=["RECOVERED"],
        knowledge_must_contain=["chi-square", "chi square", "categorical", "contingency"],
        knowledge_tags=["chi_square", "two_group"],
    ),
    _from_scenario(
        SCENARIO_BY_ID["08_two_group_fisher_exact"],
        sample_size=24,
        test_types=["fisher_exact"],
        variable_names=["COMPLICATION"],
        knowledge_must_contain=["fisher", "exact", "small sample", "expected"],
        knowledge_tags=["fisher_exact", "two_group", "small_sample"],
    ),
    _from_scenario(
        SCENARIO_BY_ID["09_multi_group_anova"],
        sample_size=90,
        test_types=["one_way_anova"],
        variable_names=["PAIN SCORE"],
        knowledge_must_contain=["anova", "analysis of variance", "three group", "multiple group"],
        knowledge_tags=["one_way_anova", "multi_group"],
    ),
    _from_scenario(
        SCENARIO_BY_ID["12_masterchart_docx"],
        sample_size=52,
        test_types=["z_test_proportion"],
        variable_names=["SEX"],
        knowledge_must_contain=["proportion", "z-test", "z test", "categorical"],
        knowledge_tags=["z_test_proportion", "single_cohort"],
    ),
    _from_scenario(
        SCENARIO_BY_ID["13_medicine5_reference"],
        sample_size=50,
        test_types=["z_test_proportion"],
        variable_names=["SEX", "FRACTURE TYPE"],
        knowledge_must_contain=["proportion", "cohort", "sample"],
        knowledge_tags=["z_test_proportion", "single_cohort"],
    ),
    _from_scenario(
        SCENARIO_BY_ID["14_complex_analysis"],
        sample_size=100,
        test_types=["unpaired_t_test", "chi_square"],
        variable_names=["PAIN SCORE", "RECOVERED", "COMPLICATION"],
        significant_variables=["PAIN SCORE", "RECOVERED"],
        knowledge_must_contain=["statistical", "test", "hypothesis", "significance"],
        knowledge_tags=["unpaired_t_test", "chi_square", "two_group"],
    ),
]

CASE_BY_ID = {c.id: c for c in RAG_EVAL_CASES}
