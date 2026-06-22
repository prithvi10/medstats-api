"""
Correction Agent - Processes professor feedback and re-runs affected pipeline stages.

The professor can provide:
1. Specific corrections (e.g., "Use paired t-test instead of unpaired for variable X")
2. Reanalysis requests (e.g., "Exclude patients under 5 years")
3. Narrative edits (e.g., "Rewrite the interpretation for Table 3")
4. Additional analysis (e.g., "Also analyze correlation between age and union time")
"""
from typing import Any, List, Dict, Optional
from src.agentic_core.agents.base_agent import BaseAgent, AgentContext, AgentPlan, AgentAction, MissionState
from src.medstats.core.stats_engine import StatsEngine
from src.shared.logging.logger import get_logger
import pandas as pd
from io import StringIO

logger = get_logger(__name__)


class CorrectionType:
    RERUN_TEST = "rerun_test"
    CHANGE_TEST = "change_test"
    EXCLUDE_DATA = "exclude_data"
    REWRITE_NARRATIVE = "rewrite_narrative"
    ADD_ANALYSIS = "add_analysis"
    GENERAL_FEEDBACK = "general_feedback"


class CorrectionAgent(BaseAgent):
    role_name = "CorrectionAgent"
    requires_llm_reasoning = False

    def __init__(self, services=None):
        super().__init__(services=services)
        self.engine = StatsEngine()

    async def _perceive_domain(self, state: MissionState, context: AgentContext) -> AgentContext:
        """Parse professor feedback and current report state."""
        raw_dataset = state.agent_outputs.get("dataframe_json", "")
        # Fallback to parsed_dataset if dataframe_json not available
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
            corrections = [self._build_correction(feedback_type, feedback, context)]
        else:
            corrections = self._classify_feedback(feedback, context)

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

        # Update report version
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

    def _classify_feedback(self, feedback: str, context: AgentContext) -> List[Dict]:
        """Classify free-text feedback into actionable corrections."""
        corrections = []
        feedback_lower = feedback.lower()

        # Detect test change requests
        test_keywords = {
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

        for keyword, test_type in test_keywords.items():
            if keyword in feedback_lower:
                # Find which variable it applies to
                target_vars = self._find_target_variables(feedback, context)
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

        # Detect exclusion requests
        exclude_words = ["exclude", "remove", "drop", "filter out", "only include"]
        if any(word in feedback_lower for word in exclude_words):
            corrections.append({
                "type": CorrectionType.EXCLUDE_DATA,
                "description": f"Apply data filter: {feedback}",
                "filter_description": feedback,
            })

        # Detect rewrite requests
        rewrite_words = ["rewrite", "rephrase", "change the wording", "edit the text"]
        if any(word in feedback_lower for word in rewrite_words):
            target_vars = self._find_target_variables(feedback, context)
            corrections.append({
                "type": CorrectionType.REWRITE_NARRATIVE,
                "description": f"Rewrite narrative for {target_vars or 'section'}",
                "target_variables": target_vars,
                "instruction": feedback,
            })

        # Detect additional analysis
        add_words = ["also analyze", "add analysis", "additionally", "also check", "also test"]
        if any(word in feedback_lower for word in add_words):
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

    def _build_correction(self, correction_type: str, feedback: str, context: AgentContext) -> Dict:
        """Build a structured correction from explicit type + feedback."""
        target_vars = context.parsed_data.get("target_variables", [])
        return {
            "type": correction_type,
            "description": f"{correction_type}: {feedback[:100]}",
            "variable": target_vars[0] if target_vars else None,
            "target_variables": target_vars,
            "instruction": feedback,
        }

    def _find_target_variables(self, feedback: str, context: AgentContext) -> List[str]:
        """Find which variables the feedback refers to."""
        variables = context.parsed_data.get("variables", [])
        targets = []
        feedback_lower = feedback.lower()

        for var_info in variables:
            var_name = var_info["name"]
            if var_name.lower() in feedback_lower:
                targets.append(var_name)

        return targets

    async def _apply_correction(
        self, correction: Dict, state: MissionState, context: AgentContext, df: Optional[pd.DataFrame]
    ) -> Dict:
        """Apply a single correction and return the result."""
        ctype = correction["type"]

        if ctype == CorrectionType.CHANGE_TEST and df is not None:
            return await self._apply_test_change(correction, state, df)
        elif ctype == CorrectionType.EXCLUDE_DATA and df is not None:
            return await self._apply_data_exclusion(correction, state, context, df)
        elif ctype == CorrectionType.RERUN_TEST and df is not None:
            return await self._apply_test_rerun(correction, state, df)
        elif ctype == CorrectionType.REWRITE_NARRATIVE:
            return await self._apply_narrative_rewrite(correction, state, context)
        elif ctype == CorrectionType.ADD_ANALYSIS and df is not None:
            return await self._apply_additional_analysis(correction, state, df)
        else:
            return {"status": "acknowledged", "type": ctype, "note": "Requires manual review"}

    async def _apply_test_change(self, correction: Dict, state: MissionState, df: pd.DataFrame) -> Dict:
        """Re-run a specific variable with a different statistical test."""
        var_name = correction.get("variable")
        new_test = correction.get("new_test")

        if not var_name or var_name not in df.columns:
            return {"status": "skipped", "type": "change_test", "reason": f"Variable '{var_name}' not found in data"}

        series = df[var_name].dropna()
        result = None

        if new_test == "z_test_proportion":
            result = self.engine.z_test_proportion(series, var_name)
        elif new_test == "chi_square":
            # For chi-square we need a grouping variable — use against the first other categorical
            variables = state.agent_outputs.get("variables", [])
            other_cat = next(
                (v["name"] for v in variables
                 if v["dtype"] == "categorical" and v["name"] != var_name and v["name"] in df.columns),
                None
            )
            if other_cat:
                result = self.engine.chi_square_test(df[other_cat].dropna(), series, var_name)
            else:
                result = self.engine.z_test_proportion(series, f"{var_name} (proportion)")
        elif new_test in ("unpaired_t_test", "mann_whitney_u"):
            # Need group info
            groups_info = state.agent_outputs.get("groups")
            if groups_info:
                group_col = groups_info.get("column")
                group_labels = groups_info.get("groups", [])
                if group_col and len(group_labels) >= 2:
                    g1 = df[df[group_col] == group_labels[0]][var_name].dropna()
                    g2 = df[df[group_col] == group_labels[1]][var_name].dropna()
                    if new_test == "unpaired_t_test":
                        result = self.engine.unpaired_t_test(g1, g2, var_name)
                    else:
                        result = self.engine.mann_whitney_u(g1, g2, var_name)

        if result:
            # Update test results in state
            test_results = state.agent_outputs.get("test_results", [])
            updated = False
            for i, tr in enumerate(test_results):
                if tr.get("variable_name") == var_name:
                    test_results[i]["test_result"] = result.to_dict()
                    updated = True
                    break
            if not updated:
                test_results.append({"variable_name": var_name, "test_result": result.to_dict()})
            state.agent_outputs["test_results"] = test_results

            self._update_report_section_test(state, var_name, result)

            return {
                "status": "applied",
                "type": "change_test",
                "variable": var_name,
                "new_test": new_test,
                "p_value": result.p_value,
            }

        return {"status": "skipped", "type": "change_test", "reason": f"Could not apply {new_test} to {var_name}"}

    async def _apply_data_exclusion(
        self, correction: Dict, state: MissionState, context: AgentContext, df: pd.DataFrame
    ) -> Dict:
        """Apply data filtering and re-run all tests on filtered data."""
        instruction = correction.get("filter_description", "")

        # Simple pattern matching for common exclusions
        original_size = len(df)
        filtered_df = df.copy()

        # Try to parse numeric filters like "exclude patients under 5 years"
        import re
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

        # Store filtered dataset
        state.agent_outputs["parsed_dataset"] = filtered_df.to_json()
        state.agent_outputs["sample_size"] = new_size
        state.agent_outputs["exclusion_note"] = (
            f"After applying filter ({instruction}), {original_size - new_size} records excluded. "
            f"Final sample: {new_size}."
        )

        return {
            "status": "applied",
            "type": "exclude_data",
            "original_size": original_size,
            "new_size": new_size,
            "excluded": original_size - new_size,
            "note": "Pipeline should be re-run on filtered data",
        }

    async def _apply_test_rerun(self, correction: Dict, state: MissionState, df: pd.DataFrame) -> Dict:
        """Re-run a test for a specific variable."""
        var_name = correction.get("variable")
        if not var_name or var_name not in df.columns:
            return {"status": "skipped", "reason": f"Variable '{var_name}' not found"}

        series = df[var_name].dropna()
        variables = state.agent_outputs.get("variables", [])
        var_info = next((v for v in variables if v["name"] == var_name), None)

        if not var_info:
            return {"status": "skipped", "reason": "Variable metadata not found"}

        dtype = var_info.get("dtype")
        if dtype == "categorical":
            result = self.engine.z_test_proportion(series, var_name)
        elif dtype == "numerical":
            normality = self.engine.shapiro_wilk(series, var_name)
            if normality.details.get("is_normal"):
                result = self.engine.compute_descriptive(series, var_name)
            else:
                result = self.engine.compute_descriptive(series, var_name)
        else:
            return {"status": "skipped", "reason": f"Unknown dtype: {dtype}"}

        return {"status": "applied", "type": "rerun_test", "variable": var_name}

    async def _apply_narrative_rewrite(
        self, correction: Dict, state: MissionState, context: AgentContext
    ) -> Dict:
        """Rewrite narrative sections using LLM or template."""
        target_vars = correction.get("target_variables", [])
        instruction = correction.get("instruction", "")

        if self.services and instruction:
            # Use LLM to rewrite
            sections = state.agent_outputs.get("report_sections", [])
            for i, section in enumerate(sections):
                if section.get("type") == "paragraph":
                    var_match = any(
                        v.lower() in section.get("content", "").lower()
                        for v in target_vars
                    ) if target_vars else True

                    if var_match:
                        try:
                            rewritten = await self.services.generate_text(
                                prompt=f"Rewrite this paragraph following this instruction: {instruction}\n\nOriginal: {section['content']}",
                                system_prompt="You are a medical biostatistician. Rewrite concisely in academic English. Keep all numbers exactly the same.",
                                temperature=0.3,
                            )
                            sections[i]["content"] = rewritten.strip()
                        except Exception:
                            pass
            state.agent_outputs["report_sections"] = sections
            return {"status": "applied", "type": "rewrite_narrative", "variables": target_vars}

        return {"status": "acknowledged", "type": "rewrite_narrative", "note": "LLM required for rewrite"}

    async def _apply_additional_analysis(
        self, correction: Dict, state: MissionState, df: pd.DataFrame
    ) -> Dict:
        """Run additional analysis on specified variables."""
        instruction = correction.get("instruction", "")
        variables = state.agent_outputs.get("variables", [])

        # Find any variables mentioned in the instruction
        target_vars = []
        for var_info in variables:
            if var_info["name"].lower() in instruction.lower():
                target_vars.append(var_info)

        if not target_vars:
            return {"status": "acknowledged", "note": "Could not identify target variables for additional analysis"}

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
