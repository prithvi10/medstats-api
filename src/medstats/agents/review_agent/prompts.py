"""
Review Agent — LLM prompts for Tier 3 interpretation checks (GPT-5).
"""

REVIEW_SYSTEM_PROMPT = """\
You are a senior biostatistics professor reviewing a postgraduate medical thesis.
Your role is to identify ONLY genuine errors, logical inconsistencies, and methodological concerns.
Do NOT flag stylistic preferences. Flag only issues that would warrant correction or re-analysis.
Be specific and actionable in your feedback.
"""

INTERPRETATION_CHECK_TEMPLATE = """\
Review this medical thesis results section for logical consistency and correct interpretation.

Study Title: {title}

Statistical Test Results (ground truth):
{test_summaries}

Narrative Excerpts (written by system):
{narratives}

Identify ONLY genuine issues in these categories:
1. Conclusions contradicted by or not supported by the statistical results
2. Claims that overstate the evidence (e.g., saying "proves" when p=0.04)
3. Important significant findings omitted from the narrative
4. Non-significant results reported as meaningful without qualification
5. Incorrect directionality of comparison (e.g., "higher in group A" when data shows group B)

Respond in JSON:
{{
  "issues": [
    {{"variable": "variable name", "issue": "what is wrong", "suggestion": "how to fix it"}}
  ],
  "overall": "ok|minor_issues|significant_issues"
}}

If no genuine issues found, return {{"issues": [], "overall": "ok"}}
"""

TEST_SELECTION_REVIEW_PROMPT = """\
Given the following variable characteristics and the statistical test that was applied, \
assess whether the test selection was appropriate.

Variable: {var_name}
Type: {var_type}
Data characteristics:
- Sample size: N={sample_size}
- Distribution: {distribution_info}
- Number of categories/groups: {n_groups}

Test applied: {test_applied}
Assumptions checked: {assumptions}

Is this test selection appropriate? If not, what test should have been used and why?

Respond in JSON:
{{
  "appropriate": true/false,
  "reason": "explanation",
  "recommended_test": "test name if inappropriate, null otherwise"
}}
"""
