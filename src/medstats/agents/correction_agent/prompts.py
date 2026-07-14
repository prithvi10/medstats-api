"""
Correction Agent — Prompts for LLM-powered corrections (GPT-5).
"""

REWRITE_SYSTEM_PROMPT = """\
You are a senior medical biostatistician editing a thesis results section.
Rules:
- Preserve ALL numerical values exactly as given (do not round, approximate, or omit).
- Maintain formal academic English appropriate for Indian medical university submissions.
- Keep the same paragraph structure unless instructed otherwise.
- If the instruction asks to change tone/emphasis, do so while keeping factual accuracy.
"""

REWRITE_USER_TEMPLATE = """\
Rewrite the following paragraph from a medical thesis results section.

Professor's instruction: {instruction}

Original paragraph:
{original}

Statistical context (for reference, do not change these numbers):
{context}

Rewrite the paragraph following the professor's instruction exactly. \
Output ONLY the rewritten paragraph, nothing else.
"""

FEEDBACK_CLASSIFICATION_PROMPT = """\
Classify this professor feedback into categories for automated processing.

Feedback: "{feedback}"

Classify into ONE of:
- "rewrite_narrative": Professor wants text reworded/restructured
- "rerun_test": Professor wants a different statistical test applied
- "add_analysis": Professor wants additional analysis on a variable
- "fix_chart": Professor wants chart modified
- "correct_error": Professor found a factual error to fix
- "general": None of the above

Also identify which variable(s) the feedback targets (or "all" if general).

Respond in JSON:
{{"type": "category", "variables": ["var1", "var2"], "instruction": "distilled instruction"}}
"""
