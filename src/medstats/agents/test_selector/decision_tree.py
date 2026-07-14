"""
Test Selector Agent — Decision tree for statistical test selection.

Maps (study_design, variable_type, normality) → appropriate test.
"""
from typing import List


SINGLE_COHORT_STEPS = ["descriptive_all", "z_test_categorical", "chi_square_associations"]
TWO_GROUP_STEPS = ["descriptive_all", "compare_groups_numerical", "compare_groups_categorical"]
MULTI_GROUP_STEPS = ["descriptive_all", "anova_numerical", "chi_square_associations"]


def get_plan_steps(study_design: str) -> List[str]:
    """Return the analysis pipeline steps for a given study design."""
    if study_design == "two_group":
        return TWO_GROUP_STEPS
    elif study_design == "multi_group":
        return MULTI_GROUP_STEPS
    return SINGLE_COHORT_STEPS
