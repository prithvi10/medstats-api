"""
Correction Agent — Processes professor feedback and re-runs affected pipeline stages.

Handles: test changes, data exclusions, narrative rewrites, additional analyses.
"""
import re
from typing import Any, List, Dict, Optional
from io import StringIO

import pandas as pd

from src.agentic_core.agents.base import BaseAgent
from src.agentic_core.agents.state import MissionState
from src.agentic_core.agents.context import AgentContext, AgentPlan, AgentAction
from src.medstats.agents.correction_agent.schemas import CorrectionType
from src.medstats.agents.correction_agent.prompts import REWRITE_SYSTEM_PROMPT, REWRITE_USER_TEMPLATE
from src.medstats.agents.correction_agent.classifier import classify_feedback, find_target_variables
from src.medstats.core.stats_engine import StatsEngine
from src.shared.logging.logger import get_logger

logger = get_logger(__name__)


class CorrectionAgent(BaseAgent):
    role_name = "CorrectionAgent"
    requires_llm_reasoning = False

    def __init__(self, services=None):
        super().__init__(services=services)
        self.engine = StatsEngine()

    async def _perceive_domain(self, state: MissionState, context: AgentContext) -> AgentContext:
        """Parse professor feedback and current report state."""
        raw_dataset = state.agent_outputs.get("dataframe_json", "")
        if not raw_dataset:
            raw_dataset = state.agent_outputs.get("parsed_dataset", "")
            if hasattr(raw_dataset, 'to_json'):
                raw_dataset = raw_dataset.to_json()
            elif not isinstance(raw_dataset, str):
                raw_dataset = ""

        context.parsed_data = {
            "feedback": state.agent_outputs.get("professor_feedback", ""),
            "feedback_type": state.agent_outputs.get("feedback_type", "general_feedback"),
            "target_variables": state.agent_outputs.get("target_variables", []),
            "current_report_sections": state.agent_outputs.get("report_sections", []),
            "current_test_results": state.agent_outputs.get("test_results", []),
            "frequency_tables": state.agent_outputs.get("frequency_tables", []),
            "dataframe_json": raw_dataset,
            "variables": state.agent_outputs.get("variables", []),
            "sample_size": state.agent_outputs.get("sample_size", 0),
            "title": state.agent_outputs.get("title", ""),
        }
        return context

    def _create_default_plan(self, state: MissionState, context: AgentContext) -> AgentPlan:
        """Classify the feedback and determine correction strategy."""
        feedback = context.parsed_data["feedback"]
        feedback_type = context.parsed_data["feedback_type"]

        if feedback_type != "general_feedback":
            target_vars = context.parsed_data.get("target_variables", [])
            corrections = [{
                "type": feedback_type,
                "description": f"{feedback_type}: {feedback[:100]}",
                "variable": target_vars[0] if target_vars else None,
                "target_variables": target_vars,
                "instruction": feedback,
            }]
        else:
            corrections = classify_feedback(feedback, context.parsed_data.get("variables", []))

        context.parsed_data["corrections"] = corrections
        return AgentPlan(
            reasoning=f"Identified {len(corrections)} correction(s) to apply",
            steps=[c["description"] for c in corrections],
        )

    async def _act_domain(self, state: MissionState, context: AgentContext, plan: AgentPlan):
        """Apply corrections to the report."""
        corrections = context.parsed_data.get("corrections", [])
        df_json = context.parsed_data.get("dataframe_json")

        df = None
        if df_json:
            df = pd.read_json(StringIO(df_json))

        applied = []
        for correction in corrections:
            result = await self._apply_correction(correction, state, context, df)
            applied.append(result)

        version = state.agent_outputs.get("report_version", 1)
        state.agent_outputs["report_version"] = version + 1
        state.agent_outputs["correction_history"] = state.agent_outputs.get("correction_history", [])
        state.agent_outputs["correction_history"].append({
            "version": version + 1,
            "feedback": context.parsed_data["feedback"],
            "corrections_applied": applied,
        })

        action = AgentAction.success_action(
            "correction",
            output=f"Applied {len(applied)} correction(s), report now at v{version + 1}",
        )
        return [action], state

    # ------------------------------------------------------------------
    # Correction dispatch
    # ------------------------------------------------------------------

    def _resolve_column(self, df: pd.DataFrame, var_name: str) -> Optional[str]:
        """Match a variable name to a dataframe column (case/space insensitive, then substring)."""
        if not var_name:
            return None
        if var_name in df.columns:
            return var_name
        target = var_name.upper().replace("_", " ").strip()
        # Exact normalised match
        for col in df.columns:
            if col.upper().replace("_", " ").strip() == target:
                return col
        # Substring match — catches abbreviations (e.g. "ROM" inside "RANGE OF MOTION")
        for col in df.columns:
            col_norm = col.upper().replace("_", " ").strip()
            if target in col_norm or col_norm in target:
                return col
        return None

    def _find_test_result_row(self, state: MissionState, var_name: str) -> Optional[int]:
        col_norm = var_name.upper().replace("_", " ").strip()
        for i, tr in enumerate(state.agent_outputs.get("test_results", [])):
            name = tr.get("variable_name", "")
            if name == var_name or name.upper().replace("_", " ").strip() == col_norm:
                return i
        return None

    async def _apply_correction(
        self, correction: Dict, state: MissionState, context: AgentContext, df: Optional[pd.DataFrame]
    ) -> Dict:
        """Apply a single correction and return the result."""
        ctype = correction["type"]

        if ctype == CorrectionType.CHANGE_TEST and df is not None:
            return await self._apply_test_change(correction, state, df)
        elif ctype == CorrectionType.EXCLUDE_DATA and df is not None:
            return await self._apply_data_exclusion(correction, state, df)
        elif ctype == CorrectionType.RERUN_TEST and df is not None:
            return await self._apply_test_rerun(correction, state, df)
        elif ctype == CorrectionType.REWRITE_NARRATIVE:
            return await self._apply_narrative_rewrite(correction, state)
        elif ctype == CorrectionType.ADD_ANALYSIS and df is not None:
            return await self._apply_additional_analysis(correction, state, df)
        else:
            return {"status": "acknowledged", "type": ctype, "note": "Requires manual review"}

    async def _apply_test_change(self, correction: Dict, state: MissionState, df: pd.DataFrame) -> Dict:
        """Re-run a specific variable with a different statistical test."""
        var_name = correction.get("variable")
        new_test = correction.get("new_test")
        col = self._resolve_column(df, var_name) if var_name else None

        if not col:
            return {"status": "skipped", "type": "change_test", "reason": f"Variable '{var_name}' not found"}

        series = df[col].dropna()
        result = None
        variables = state.agent_outputs.get("variables", [])
        groups_info = state.agent_outputs.get("groups")
        study_design = state.agent_outputs.get("study_design", "single_cohort")

        if new_test == "z_test_proportion":
            result = self.engine.z_test_proportion(series, col)
        elif new_test == "chi_square":
            other_cat = next(
                (v["name"] for v in variables
                 if v.get("dtype") == "categorical" and self._resolve_column(df, v["name"]) != col
                 and self._resolve_column(df, v["name"]) in df.columns),
                None,
            )
            if other_cat and groups_info and study_design != "single_cohort":
                group_col = groups_info.get("column")
                if group_col and group_col in df.columns:
                    result = self.engine.chi_square_test(df[group_col], series, col)
            elif other_cat:
                other_col = self._resolve_column(df, other_cat)
                result = self.engine.chi_square_test(df[other_col].dropna(), series, col)
            else:
                result = self.engine.z_test_proportion(series, col)
        elif new_test == "fisher_exact":
            other_cat = next(
                (v["name"] for v in variables
                 if v.get("dtype") == "categorical" and self._resolve_column(df, v["name"]) != col
                 and self._resolve_column(df, v["name"]) in df.columns),
                None,
            )
            if other_cat and groups_info:
                group_col = groups_info.get("column")
                if group_col and group_col in df.columns:
                    result = self.engine.chi_square_test(df[group_col], series, col)
                else:
                    other_col = self._resolve_column(df, other_cat)
                    result = self.engine.fisher_exact_test(df[other_col].dropna(), series, col)
            else:
                # Single-cohort fallback: no group column available for 2×2 table
                logger.info(
                    "fisher_exact requested but no group column available; "
                    "falling back to z_test_proportion",
                    variable=col,
                )
                result = self.engine.z_test_proportion(series, col)
        elif new_test in ("unpaired_t_test", "mann_whitney_u", "paired_t_test"):
            if groups_info:
                group_col = groups_info.get("column")
                group_labels = groups_info.get("groups", [])
                if group_col and group_col in df.columns and len(group_labels) >= 2:
                    g1 = df[df[group_col] == group_labels[0]][col].dropna()
                    g2 = df[df[group_col] == group_labels[1]][col].dropna()
                    if new_test == "unpaired_t_test":
                        result = self.engine.unpaired_t_test(g1, g2, col)
                    elif new_test == "mann_whitney_u":
                        result = self.engine.mann_whitney_u(g1, g2, col)
                    elif new_test == "paired_t_test" and len(g1) == len(g2):
                        result = self.engine.paired_t_test(g1, g2, col)
        elif new_test in ("one_way_anova", "kruskal_wallis"):
            if groups_info:
                group_col = groups_info.get("column")
                group_labels = groups_info.get("groups") or []
                if group_col and group_col in df.columns and len(group_labels) >= 3:
                    group_series = [df[df[group_col] == label][col].dropna() for label in group_labels]
                    if new_test == "one_way_anova":
                        result = self.engine.one_way_anova(group_series, col)
                    else:
                        result = self.engine.kruskal_wallis(group_series, col)

        if result:
            test_results = state.agent_outputs.get("test_results", [])
            row_idx = self._find_test_result_row(state, col)
            payload = result.to_dict()
            if row_idx is not None:
                test_results[row_idx]["test_result"] = payload
                test_results[row_idx]["variable_name"] = col
            else:
                test_results.append({"variable_name": col, "test_result": payload})
            state.agent_outputs["test_results"] = test_results
            self._update_report_section_test(state, col, result)
            return {
                "status": "applied",
                "type": "change_test",
                "variable": col,
                "new_test": new_test,
                "p_value": result.p_value,
            }

        return {"status": "skipped", "type": "change_test", "reason": f"Could not apply {new_test} to {col}"}

    async def _apply_data_exclusion(self, correction: Dict, state: MissionState, df: pd.DataFrame) -> Dict:
        """Apply data filtering and re-run all tests on filtered data."""
        instruction = correction.get("filter_description", "")
        original_size = len(df)
        filtered_df = df.copy()

        age_match = re.search(r'(?:under|below|less than|<)\s*(\d+)\s*(?:years?|yrs?)?', instruction.lower())
        if age_match and "AGE" in df.columns:
            threshold = int(age_match.group(1))
            filtered_df = filtered_df[filtered_df["AGE"] >= threshold]

        age_match_above = re.search(r'(?:over|above|greater than|>)\s*(\d+)\s*(?:years?|yrs?)?', instruction.lower())
        if age_match_above and "AGE" in df.columns:
            threshold = int(age_match_above.group(1))
            filtered_df = filtered_df[filtered_df["AGE"] <= threshold]

        new_size = len(filtered_df)
        if new_size == original_size:
            return {"status": "no_change", "reason": "Filter did not exclude any records"}

        state.agent_outputs["parsed_dataset"] = filtered_df.to_json()
        state.agent_outputs["sample_size"] = new_size
        state.agent_outputs["exclusion_note"] = (
            f"After applying filter ({instruction}), {original_size - new_size} records excluded. "
            f"Final sample: {new_size}."
        )

        return {
            "status": "applied", "type": "exclude_data",
            "original_size": original_size, "new_size": new_size,
            "excluded": original_size - new_size,
            "note": "Pipeline should be re-run on filtered data",
        }

    async def _apply_test_rerun(self, correction: Dict, state: MissionState, df: pd.DataFrame) -> Dict:
        """Re-run a test for a specific variable and persist results."""
        var_name = correction.get("variable")
        col = self._resolve_column(df, var_name) if var_name else None
        if not col:
            return {"status": "skipped", "reason": f"Variable '{var_name}' not found"}

        series = df[col].dropna()
        variables = state.agent_outputs.get("variables", [])
        var_info = next(
            (v for v in variables if self._resolve_column(df, v.get("name", "")) == col or v.get("name") == var_name),
            None,
        )

        if not var_info:
            return {"status": "skipped", "reason": "Variable metadata not found"}

        dtype = var_info.get("dtype")
        result = None
        if dtype == "categorical":
            result = self.engine.z_test_proportion(series, col)
        elif dtype == "numerical":
            result = self.engine.shapiro_wilk(series, col)

        if not result:
            return {"status": "skipped", "reason": f"Could not rerun test for {col}"}

        test_results = state.agent_outputs.get("test_results", [])
        row_idx = self._find_test_result_row(state, col)
        payload = result.to_dict()
        if row_idx is not None:
            if dtype == "numerical":
                # Append normality as supplementary row if not main test_result
                existing = test_results[row_idx].get("test_result")
                if existing and existing.get("test_type") not in (None, "shapiro_wilk"):
                    test_results.append({"variable_name": col, "test_result": payload})
                else:
                    test_results[row_idx]["test_result"] = payload
            else:
                test_results[row_idx]["test_result"] = payload
        else:
            test_results.append({"variable_name": col, "test_result": payload})
        state.agent_outputs["test_results"] = test_results
        return {"status": "applied", "type": "rerun_test", "variable": col, "test_type": payload.get("test_type")}

    async def _apply_narrative_rewrite(self, correction: Dict, state: MissionState) -> Dict:
        """Rewrite narrative sections using LLM or template fallback."""
        target_vars = correction.get("target_variables", [])
        instruction = correction.get("instruction", "")
        scope = correction.get("scope", "variable")

        sections = state.agent_outputs.get("report_sections", [])

        if scope == "methodology":
            for i, section in enumerate(sections):
                if section.get("type") == "heading" and "statistical analysis" in (section.get("content") or "").lower():
                    if i + 1 < len(sections) and sections[i + 1].get("type") == "paragraph":
                        original = sections[i + 1].get("content", "")
                        sections[i + 1]["content"] = self._template_rewrite(original, instruction)
                        state.agent_outputs["report_sections"] = sections
                        return {"status": "applied", "type": "rewrite_narrative", "scope": "methodology"}
            return {"status": "skipped", "type": "rewrite_narrative", "reason": "Methodology section not found"}

        if self.services and instruction:
            test_results = state.agent_outputs.get("test_results", [])

            for i, section in enumerate(sections):
                if section.get("type") not in ("paragraph", "summary_text"):
                    continue
                var_match = any(
                    v.lower() in (section.get("content") or "").lower() for v in target_vars
                ) if target_vars else scope != "variable"

                if var_match:
                    context_parts = []
                    for tr in test_results:
                        if not target_vars or any(
                            v.lower() in tr.get("variable_name", "").lower() for v in target_vars
                        ):
                            inner = tr.get("test_result", {})
                            if isinstance(inner, dict) and inner.get("significance_statement"):
                                context_parts.append(f"{tr['variable_name']}: {inner['significance_statement']}")

                    try:
                        prompt = REWRITE_USER_TEMPLATE.format(
                            instruction=instruction,
                            original=section['content'],
                            context="\n".join(context_parts) if context_parts else "N/A",
                        )
                        rewritten = await self.services.generate_text(
                            prompt=prompt,
                            system_prompt=REWRITE_SYSTEM_PROMPT,
                            temperature=0.3,
                        )
                        sections[i]["content"] = rewritten.strip()
                    except Exception as e:
                        logger.warning("Narrative rewrite failed", error=str(e))
                        sections[i]["content"] = self._template_rewrite(section.get("content", ""), instruction)
            state.agent_outputs["report_sections"] = sections
            return {"status": "applied", "type": "rewrite_narrative", "variables": target_vars}

        if instruction and sections:
            for i, section in enumerate(sections):
                if section.get("type") in ("paragraph", "summary_text"):
                    if not target_vars or any(
                        v.lower() in (section.get("content") or "").lower() for v in target_vars
                    ):
                        sections[i]["content"] = self._template_rewrite(section.get("content", ""), instruction)
            state.agent_outputs["report_sections"] = sections
            return {"status": "applied", "type": "rewrite_narrative", "note": "template rewrite"}

        return {"status": "acknowledged", "type": "rewrite_narrative", "note": "No sections to rewrite"}

    def _template_rewrite(self, original: str, instruction: str) -> str:
        if not original:
            return instruction
        if instruction.lower() in original.lower():
            return original
        return f"{original.rstrip()} {instruction}".strip()

    async def _apply_additional_analysis(self, correction: Dict, state: MissionState, df: pd.DataFrame) -> Dict:
        """Run additional analysis on specified variables."""
        instruction = correction.get("instruction", "")
        variables = state.agent_outputs.get("variables", [])

        target_vars = [v for v in variables if v["name"].lower() in instruction.lower()]
        if not target_vars:
            return {"status": "acknowledged", "note": "Could not identify target variables"}

        new_results = []
        for var_info in target_vars:
            var_name = var_info["name"]
            if var_name not in df.columns:
                continue
            series = df[var_name].dropna()
            desc = self.engine.compute_descriptive(series, var_name)
            new_results.append({"variable_name": var_name, "descriptive": desc.__dict__})

        if new_results:
            existing = state.agent_outputs.get("test_results", [])
            existing.extend(new_results)
            state.agent_outputs["test_results"] = existing

        return {"status": "applied", "type": "add_analysis", "new_variables": [v["name"] for v in target_vars]}

    def _update_report_section_test(self, state: MissionState, var_name: str, result) -> None:
        """Update the test result text in report sections for a specific variable."""
        sections = state.agent_outputs.get("report_sections", [])
        found_var_section = False

        for i, section in enumerate(sections):
            if section.get("variable_name") == var_name:
                found_var_section = True
            elif found_var_section and section.get("type") == "test_result":
                sections[i]["content"] = result.significance_statement
                found_var_section = False
                break

        state.agent_outputs["report_sections"] = sections
