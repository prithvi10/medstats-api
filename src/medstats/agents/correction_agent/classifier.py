"""
Correction Agent — Feedback classifier.

Classifies free-text professor feedback into structured correction actions.
"""
from typing import List, Dict

from src.medstats.agents.correction_agent.schemas import CorrectionType


TEST_KEYWORDS = {
    "t-test": "unpaired_t_test",
    "paired t": "paired_t_test",
    "chi-square": "chi_square",
    "chi square": "chi_square",
    "fisher": "fisher_exact",
    "mann-whitney": "mann_whitney_u",
    "anova": "one_way_anova",
    "kruskal": "kruskal_wallis",
    "z-test": "z_test_proportion",
}

EXCLUDE_KEYWORDS = ["exclude", "remove", "drop", "filter out", "only include"]
REWRITE_KEYWORDS = ["rewrite", "rephrase", "change the wording", "edit the text"]
ADD_ANALYSIS_KEYWORDS = ["also analyze", "add analysis", "additionally", "also check", "also test"]


def classify_feedback(feedback: str, variables: List[Dict]) -> List[Dict]:
    """Classify free-text feedback into actionable corrections."""
    corrections = []
    feedback_lower = feedback.lower()

    for keyword, test_type in TEST_KEYWORDS.items():
        if keyword in feedback_lower:
            target_vars = find_target_variables(feedback, variables)
            for var in target_vars:
                corrections.append({
                    "type": CorrectionType.CHANGE_TEST,
                    "description": f"Change test for {var} to {test_type}",
                    "variable": var,
                    "new_test": test_type,
                })
            if not target_vars:
                corrections.append({
                    "type": CorrectionType.CHANGE_TEST,
                    "description": f"Change test to {test_type}",
                    "variable": None,
                    "new_test": test_type,
                })

    if any(word in feedback_lower for word in EXCLUDE_KEYWORDS):
        corrections.append({
            "type": CorrectionType.EXCLUDE_DATA,
            "description": f"Apply data filter: {feedback}",
            "filter_description": feedback,
        })

    if any(word in feedback_lower for word in REWRITE_KEYWORDS):
        target_vars = find_target_variables(feedback, variables)
        corrections.append({
            "type": CorrectionType.REWRITE_NARRATIVE,
            "description": f"Rewrite narrative for {target_vars or 'section'}",
            "target_variables": target_vars,
            "instruction": feedback,
        })

    if any(word in feedback_lower for word in ADD_ANALYSIS_KEYWORDS):
        corrections.append({
            "type": CorrectionType.ADD_ANALYSIS,
            "description": f"Add analysis: {feedback}",
            "instruction": feedback,
        })

    if not corrections:
        corrections.append({
            "type": CorrectionType.GENERAL_FEEDBACK,
            "description": f"General feedback: {feedback[:100]}",
            "feedback": feedback,
        })

    return corrections


def find_target_variables(feedback: str, variables: List[Dict]) -> List[str]:
    """Find which variables the feedback refers to."""
    targets = []
    feedback_lower = feedback.lower()

    for var_info in variables:
        var_name = var_info["name"]
        if var_name.lower() in feedback_lower:
            targets.append(var_name)

    return targets
