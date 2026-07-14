"""
Revision Resolver Agent — Processes structured professor reviews.

Applies statistical corrections and regenerates report, summary, methodology, and charts.
"""
from typing import Any, Dict, List, Optional

import pandas as pd

from src.agentic_core.agents.base import BaseAgent
from src.agentic_core.agents.state import MissionState
from src.agentic_core.agents.context import AgentContext, AgentPlan, AgentAction
from src.medstats.agents.correction_agent.schemas import CorrectionType
from src.medstats.agents.correction_agent.agent import CorrectionAgent
from src.medstats.agents.report_generator import ReportGeneratorAgent
from src.medstats.agents.report_generator.formatters import generate_methodology_note
from src.medstats.services.review_presets import PRESET_INSTRUCTIONS
from src.shared.logging.logger import get_logger

logger = get_logger(__name__)

REWRITE_CODES = {"unclear_wording", "strengthen_conclusion", "interpretation_overreach", "add_limitations"}
METHODOLOGY_RERUN_CODES = {"missing_normality", "wrong_test", "study_design_mismatch"}
TEST_CHANGE_CODES = {"missing_normality", "wrong_test", "study_design_mismatch"}
# Codes that trigger a narrative rewrite at the variable level
NARRATIVE_CODES = {"add_ci", "clarify_groups", "rerun_descriptive", "unclear_wording", "strengthen_conclusion", "interpretation_overreach", "add_limitations"}
PARAMETRIC_ALTERNATIVES = {
    "unpaired_t_test": "mann_whitney_u",
    "paired_t_test": "mann_whitney_u",
    "one_way_anova": "kruskal_wallis",
    "chi_square": "fisher_exact",
}


class RevisionResolverAgent(BaseAgent):
    role_name = "RevisionResolver"
    requires_llm_reasoning = False

    def __init__(self, services=None):
        super().__init__(services=services)
        self._correction_helper = CorrectionAgent(services=services)

    async def _perceive_domain(self, state: MissionState, context: AgentContext) -> AgentContext:
        structured = state.agent_outputs.get("structured_review", {})
        context.parsed_data = {
            "structured_review": structured,
            "corrections": self._build_corrections(structured, state.agent_outputs),
            "variables": state.agent_outputs.get("variables", []),
            "dataframe_json": state.agent_outputs.get("dataframe_json", ""),
        }
        return context

    def _create_default_plan(self, state: MissionState, context: AgentContext) -> AgentPlan:
        corrections = context.parsed_data.get("corrections", [])
        context.parsed_data["corrections"] = corrections
        return AgentPlan(
            reasoning=f"Applying {len(corrections)} structured correction(s) from professor review",
            steps=[c["description"] for c in corrections] or ["Regenerate report with professor feedback"],
        )

    async def _act_domain(self, state: MissionState, context: AgentContext, plan: AgentPlan):
        from io import StringIO

        corrections = context.parsed_data.get("corrections", [])
        structured = context.parsed_data.get("structured_review", {})
        df_json = context.parsed_data.get("dataframe_json")
        df = pd.read_json(StringIO(df_json)) if df_json else None

        applied = []
        for correction in corrections:
            result = await self._correction_helper._apply_correction(correction, state, context, df)
            applied.append(result)

        reporter = ReportGeneratorAgent(services=self.services)
        state = await reporter.run(state)

        self._refresh_methodology_section(state)
        await self._refresh_executive_summary(state, structured)

        version = state.agent_outputs.get("report_version", 1)
        state.agent_outputs["report_version"] = version + 1
        state.agent_outputs["revision_history"] = state.agent_outputs.get("revision_history", [])
        state.agent_outputs["revision_history"].append({
            "version": version + 1,
            "structured_review": structured,
            "corrections_applied": applied,
        })

        action = AgentAction.success_action(
            "revision_resolver",
            output=f"Resolved professor review; report now at v{version + 1}",
        )
        return [action], state

    def _current_test(self, test_results: List[Dict], var_name: str) -> Optional[str]:
        col_norm = var_name.upper().replace("_", " ").strip()
        for tr in test_results:
            name = tr.get("variable_name", "")
            if name != var_name and name.upper().replace("_", " ").strip() != col_norm:
                continue
            inner = tr.get("test_result") or {}
            return inner.get("test_type")
        return None

    def _variable_dtype(self, variables: List[Dict], var_name: str) -> Optional[str]:
        col_norm = var_name.upper().replace("_", " ").strip()
        for var in variables:
            name = var.get("name", "")
            if name == var_name or name.upper().replace("_", " ").strip() == col_norm:
                return var.get("dtype")
        return None

    def _suggest_alternative_test(
        self,
        current_test: Optional[str],
        dtype: Optional[str],
        codes: List[str],
    ) -> Optional[str]:
        if not current_test or not any(c in TEST_CHANGE_CODES for c in codes):
            return None
        if dtype == "numerical" and current_test in PARAMETRIC_ALTERNATIVES:
            return PARAMETRIC_ALTERNATIVES[current_test]
        if dtype == "categorical" and current_test == "chi_square" and "wrong_test" in codes:
            return "fisher_exact"
        return None

    def _append_test_change(
        self,
        corrections: List[Dict],
        scheduled: set,
        var_name: str,
        new_test: str,
        reason: str,
    ) -> None:
        key = (var_name, new_test)
        if key in scheduled:
            return
        scheduled.add(key)
        corrections.append({
            "type": CorrectionType.CHANGE_TEST,
            "description": reason,
            "variable": var_name,
            "new_test": new_test,
        })

    def _build_corrections(self, structured: Dict[str, Any], agent_outputs: Dict[str, Any]) -> List[Dict]:
        corrections = []
        variables = agent_outputs.get("variables", [])
        test_results = agent_outputs.get("test_results", [])
        scheduled_changes: set = set()
        overrides_by_var = {
            o["variable"]: o["new_test"]
            for o in structured.get("test_overrides", [])
            if o.get("variable") and o.get("new_test")
        }

        for override in structured.get("test_overrides", []):
            var_name = override.get("variable")
            new_test = override.get("new_test")
            if var_name and new_test:
                self._append_test_change(
                    corrections,
                    scheduled_changes,
                    var_name,
                    new_test,
                    f"Professor override: {var_name} → {new_test}",
                )

        for feedback in structured.get("section_feedback", []):
            target = feedback.get("target", "")
            codes = feedback.get("preset_codes", [])
            if not codes:
                continue

            if target in ("methodology", "executive_summary"):
                if any(c in REWRITE_CODES for c in codes):
                    corrections.append({
                        "type": CorrectionType.REWRITE_NARRATIVE,
                        "description": f"Revise {target}: {', '.join(codes)}",
                        "target_variables": [],
                        "instruction": " ".join(PRESET_INSTRUCTIONS.get(c, c) for c in codes),
                        "scope": target,
                    })
                if target == "methodology":
                    if any(c in TEST_CHANGE_CODES for c in codes):
                        for var in variables:
                            var_name = var.get("name")
                            if not var_name or var_name in overrides_by_var:
                                continue
                            current = self._current_test(test_results, var_name)
                            alt = self._suggest_alternative_test(current, var.get("dtype"), codes)
                            if alt and alt != current:
                                self._append_test_change(
                                    corrections,
                                    scheduled_changes,
                                    var_name,
                                    alt,
                                    f"Address methodology feedback ({', '.join(codes)}): {var_name} → {alt}",
                                )
                    if "missing_normality" in codes:
                        for var in variables:
                            if var.get("dtype") == "numerical":
                                corrections.append({
                                    "type": CorrectionType.RERUN_TEST,
                                    "description": f"Add normality check for {var['name']}",
                                    "variable": var["name"],
                                })
                continue

            # Per-variable feedback
            if any(c in TEST_CHANGE_CODES for c in codes) and target not in overrides_by_var:
                current = self._current_test(test_results, target)
                dtype = self._variable_dtype(variables, target)
                alt = self._suggest_alternative_test(current, dtype, codes)
                if alt and alt != current:
                    self._append_test_change(
                        corrections,
                        scheduled_changes,
                        target,
                        alt,
                        f"Address variable feedback ({', '.join(codes)}): {target} → {alt}",
                    )

            if "rerun_descriptive" in codes:
                corrections.append({
                    "type": CorrectionType.RERUN_TEST,
                    "description": f"Rerun descriptive for {target}",
                    "variable": target,
                })
            if "missing_normality" in codes:
                corrections.append({
                    "type": CorrectionType.RERUN_TEST,
                    "description": f"Add normality check for {target}",
                    "variable": target,
                })
            # Any narrative-level code produces a rewrite (add_ci, clarify_groups, etc.)
            narrative_codes = [c for c in codes if c in NARRATIVE_CODES and c != "rerun_descriptive"]
            if narrative_codes:
                corrections.append({
                    "type": CorrectionType.REWRITE_NARRATIVE,
                    "description": f"Revise narrative for {target}: {', '.join(narrative_codes)}",
                    "target_variables": [target],
                    "instruction": " ".join(PRESET_INSTRUCTIONS.get(c, c) for c in narrative_codes),
                    "scope": "variable",
                })

        final_comment = (structured.get("final_comment") or "").strip()
        if final_comment:
            corrections.append({
                "type": CorrectionType.REWRITE_NARRATIVE,
                "description": "Professor final comment",
                "target_variables": [],
                "instruction": final_comment,
                "scope": "executive_summary",
            })

        return corrections

    def _collect_tests_used(self, test_results: List[Dict]) -> set:
        tests_used = set()
        for row in test_results:
            inner = row.get("test_result")
            if isinstance(inner, dict) and inner.get("test_type"):
                tests_used.add(inner["test_type"])
        return tests_used

    def _refresh_methodology_section(self, state: MissionState) -> None:
        """Rebuild methodology paragraph from updated test results — no professor-note injection."""
        test_results = state.agent_outputs.get("test_results", [])
        tests_used = self._collect_tests_used(test_results)
        sample_size = state.agent_outputs.get("sample_size", 0)
        methodology = generate_methodology_note(tests_used, sample_size)
        state.agent_outputs["methodology_note"] = methodology

        sections = list(state.agent_outputs.get("report_sections", []))
        for i, section in enumerate(sections):
            if section.get("type") != "heading":
                continue
            if "statistical analysis" not in (section.get("content") or "").lower():
                continue
            if i + 1 < len(sections) and sections[i + 1].get("type") == "paragraph":
                sections[i + 1]["content"] = methodology
            else:
                sections.insert(i + 1, {"type": "paragraph", "content": methodology})
            break
        state.agent_outputs["report_sections"] = sections

    async def _refresh_executive_summary(self, state: MissionState, structured: Dict[str, Any]) -> None:
        """Regenerate executive summary from updated test results and apply HITL hints."""
        reporter = ReportGeneratorAgent(services=self.services)
        test_results = state.agent_outputs.get("test_results", [])
        title = state.agent_outputs.get("title", "")
        sample_size = state.agent_outputs.get("sample_size", 0)

        summary = await reporter._generate_summary(test_results, title, sample_size)

        exec_codes = []
        for fb in structured.get("section_feedback", []):
            if fb.get("target") == "executive_summary":
                exec_codes.extend(fb.get("preset_codes", []))

        if exec_codes or structured.get("final_comment"):
            summary = self._apply_summary_hints(summary, exec_codes, structured.get("final_comment"))

        state.agent_outputs["summary"] = summary

    def _apply_summary_hints(self, summary: str, codes: List[str], final_comment: Optional[str]) -> str:
        import json

        try:
            data = json.loads(summary)
        except json.JSONDecodeError:
            data = {"study_description": summary, "key_findings": [], "methodology": ""}

        if "add_limitations" in codes:
            limitation = (
                "Study limitations include sample size constraints and the observational design, "
                "which limit causal inference."
            )
            data["study_description"] = (data.get("study_description") or "") + " " + limitation

        if "strengthen_conclusion" in codes:
            sig = [f for f in data.get("key_findings", []) if f.get("significant")]
            if sig:
                data["study_description"] = (
                    (data.get("study_description") or "")
                    + f" A key conclusion is that {sig[0].get('variable')} showed a statistically significant finding."
                )

        if "cite_assumptions" in codes:
            data["methodology"] = (
                (data.get("methodology") or "")
                + " Statistical assumptions including normality and independence were considered when selecting tests."
            )

        if final_comment:
            data["study_description"] = (data.get("study_description") or "") + " " + final_comment

        return json.dumps(data)
