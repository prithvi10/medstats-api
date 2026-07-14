"""
Correction Agent — Schemas and correction types.
"""


class CorrectionType:
    """Types of corrections that can be applied."""

    RERUN_TEST = "rerun_test"
    CHANGE_TEST = "change_test"
    EXCLUDE_DATA = "exclude_data"
    REWRITE_NARRATIVE = "rewrite_narrative"
    ADD_ANALYSIS = "add_analysis"
    GENERAL_FEEDBACK = "general_feedback"


REWRITE_SYSTEM_PROMPT = (
    "You are a medical biostatistician. "
    "Rewrite concisely in academic English. Keep all numbers exactly the same."
)
