"""
Test Selector Agent - Selects appropriate statistical tests based on
study design, variable types, and sample size.
"""
import pandas as pd
from typing import Any, List, Dict, Tuple
from dataclasses import dataclass, field
from src.agentic_core.agents.base_agent import BaseAgent, AgentContext, AgentPlan, AgentAction, MissionState
from src.medstats.core.stats_engine import StatsEngine, TestResult, TestType
from src.shared.logging.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TestSelection:
    variable_name: str
    test_type: TestType
    justification: str
    params: Dict[str, Any] = field(default_factory=dict)


class TestSelectorAgent(BaseAgent):
    role_name = "TestSelector"
    requires_llm_reasoning = False

    def __init__(self, services=None):
        super().__init__(services=services)
        self.engine = StatsEngine()

    async def _perceive_domain(self, state: MissionState, context: AgentContext) -> AgentContext:
        """Load parsed dataset from previous agent's output."""
        context.parsed_data = {
            "variables": state.agent_outputs.get("variables", []),
            "study_design": state.agent_outputs.get("study_design", "single_cohort"),
            "groups": state.agent_outputs.get("groups"),
            "sample_size": state.agent_outputs.get("sample_size", 0),
            "dataframe_json": state.agent_outputs.get("dataframe_json"),
        }
        return context

    def _create_default_plan(self, state: MissionState, context: AgentContext) -> AgentPlan:
        """Deterministic test selection based on variable types and study design."""
        study_design = context.parsed_data["study_design"]
        variables = context.parsed_data["variables"]

        steps = []
        if study_design == "single_cohort":
            steps = ["descriptive_all", "z_test_categorical", "chi_square_associations"]
        elif study_design == "two_group":
            steps = ["descriptive_all", "compare_groups_numerical", "compare_groups_categorical"]
        elif study_design == "multi_group":
            steps = ["descriptive_all", "anova_numerical", "chi_square_associations"]

        return AgentPlan(
            steps=steps,
            selected_tools=["stats_engine"],
            confidence=0.9,
            reasoning=f"Study design: {study_design}, {len(variables)} variables",
        )

    async def _act_domain(self, state: MissionState, context: AgentContext, plan: AgentPlan):
        """Run selected tests and produce results."""
        df_json = context.parsed_data.get("dataframe_json")
        if not df_json:
            return [AgentAction.failure_action("test_select", "No dataframe available")], state

        from io import StringIO
        df = pd.read_json(StringIO(df_json))
        variables = context.parsed_data["variables"]
        study_design = context.parsed_data["study_design"]
        groups_info = context.parsed_data.get("groups")

        all_results: List[Dict] = []
        all_tables: List[Dict] = []

        for var_info in variables:
            var_name = var_info["name"]
            if var_name not in df.columns:
                continue

            series = df[var_name]
            dtype = var_info["dtype"]

            # Skip columns that can't be processed
            if dtype == "numerical":
                try:
                    series = pd.to_numeric(series, errors="coerce")
                    if series.dropna().empty:
                        continue
                except (ValueError, TypeError):
                    continue

            # Descriptive stats + frequency table for all variables
            descriptive = self.engine.compute_descriptive(series, var_name)
            freq_table = self.engine.compute_frequency_table(series, var_name)
            all_tables.append(freq_table)

            # Select and run appropriate test
            test_result = None

            if study_design == "single_cohort":
                if dtype == "categorical":
                    test_result = self.engine.z_test_proportion(series, var_name)
                elif dtype == "numerical":
                    normality = self.engine.shapiro_wilk(series, var_name)
                    all_results.append(normality.to_dict())

            elif study_design == "two_group" and groups_info:
                group_col = groups_info.get("column")
                if group_col and group_col in df.columns:
                    group_labels = groups_info.get("groups", [])
                    if len(group_labels) >= 2:
                        g1 = df[df[group_col] == group_labels[0]][var_name]
                        g2 = df[df[group_col] == group_labels[1]][var_name]

                        if dtype == "numerical":
                            normality = self.engine.shapiro_wilk(series, var_name)
                            if normality.details.get("is_normal", True):
                                test_result = self.engine.unpaired_t_test(g1, g2, var_name)
                            else:
                                test_result = self.engine.mann_whitney_u(g1, g2, var_name)
                        elif dtype == "categorical":
                            test_result = self.engine.chi_square_test(
                                df[group_col], series, var_name
                            )

            if test_result:
                all_results.append(test_result.to_dict())

            # Store per-variable result bundle
            all_results.append({
                "variable_name": var_name,
                "dtype": dtype,
                "descriptive": vars(descriptive),
                "frequency_table": freq_table,
                "test_result": test_result.to_dict() if test_result else None,
            })

        state.agent_outputs["test_results"] = all_results
        state.agent_outputs["frequency_tables"] = all_tables
        state.agent_outputs["tests_run_count"] = len([r for r in all_results if r.get("test_result")])

        action = AgentAction.success_action(
            "test_selection",
            output=f"Ran tests on {len(variables)} variables, {len(all_results)} results",
        )
        return [action], state
