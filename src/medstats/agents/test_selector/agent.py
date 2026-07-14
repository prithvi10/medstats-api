"""
Test Selector Agent — Selects appropriate statistical tests based on
study design, variable types, and sample size.
"""
from typing import Any, List, Dict
from io import StringIO

import pandas as pd

from src.agentic_core.agents.base import BaseAgent
from src.agentic_core.agents.state import MissionState
from src.agentic_core.agents.context import AgentContext, AgentPlan, AgentAction
from src.medstats.agents.test_selector.decision_tree import get_plan_steps
from src.medstats.core.stats_engine import StatsEngine
from src.medstats.agents.rag_knowledge import get_rag_agent
from src.shared.logging.logger import get_logger

logger = get_logger(__name__)


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
        steps = get_plan_steps(study_design)

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

            if dtype == "numerical":
                try:
                    series = pd.to_numeric(series, errors="coerce")
                    if series.dropna().empty:
                        continue
                except (ValueError, TypeError):
                    continue

            descriptive = self.engine.compute_descriptive(series, var_name)
            freq_table = self.engine.compute_frequency_table(series, var_name)
            all_tables.append(freq_table)

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
                            all_results.append(normality.to_dict())
                            if normality.details.get("is_normal", True):
                                test_result = self.engine.unpaired_t_test(g1, g2, var_name)
                            else:
                                test_result = self.engine.mann_whitney_u(g1, g2, var_name)
                        elif dtype == "categorical":
                            test_result = self.engine.chi_square_test(
                                df[group_col], series, var_name
                            )

            elif study_design == "multi_group" and groups_info:
                group_col = groups_info.get("column")
                group_labels = groups_info.get("groups") or []
                if group_col and group_col in df.columns and len(group_labels) >= 3:
                    group_series = [
                        df[df[group_col] == label][var_name] for label in group_labels
                    ]
                    if dtype == "numerical":
                        normality = self.engine.shapiro_wilk(series, var_name)
                        all_results.append(normality.to_dict())
                        if normality.details.get("is_normal", True):
                            test_result = self.engine.one_way_anova(group_series, var_name)
                        else:
                            test_result = self.engine.kruskal_wallis(group_series, var_name)
                    elif dtype == "categorical":
                        test_result = self.engine.chi_square_test(
                            df[group_col], series, var_name
                        )

            raw_values = None
            if dtype == "numerical":
                numeric = pd.to_numeric(series, errors="coerce").dropna()
                if len(numeric) >= 10:
                    raw_values = numeric.astype(float).tolist()

            if test_result:
                all_results.append(test_result.to_dict())

            all_results.append({
                "variable_name": var_name,
                "dtype": dtype,
                "descriptive": vars(descriptive),
                "frequency_table": freq_table,
                "test_result": test_result.to_dict() if test_result else None,
                "raw_values": raw_values,
            })

        state.agent_outputs["test_results"] = all_results
        state.agent_outputs["frequency_tables"] = all_tables
        state.agent_outputs["tests_run_count"] = len([r for r in all_results if r.get("test_result")])

        rag_rationale = await self._fetch_rag_test_rationale(all_results, study_design)
        if rag_rationale:
            state.agent_outputs["rag_test_rationale"] = rag_rationale

        action = AgentAction.success_action(
            "test_selection",
            output=f"Ran tests on {len(variables)} variables, {len(all_results)} results",
        )
        return [action], state

    async def _fetch_rag_test_rationale(
        self, all_results: List[Dict], study_design: str
    ) -> Dict[str, List[str]]:
        """Retrieve grounded guideline passages for each test used."""
        rag_agent = get_rag_agent()
        if not rag_agent:
            return {}

        rationale: Dict[str, List[str]] = {}
        seen: set = set()

        for row in all_results:
            tr = row.get("test_result")
            if not tr or not isinstance(tr, dict):
                continue
            test_type = tr.get("test_type")
            if not test_type or test_type in seen:
                continue
            seen.add(test_type)

            result = await rag_agent.retrieve(
                query=f"when and why to use {test_type} in a {study_design} medical study",
                tags=[test_type, study_design],
                top_k=2,
            )
            if result.passages:
                rationale[test_type] = result.passages

        return rationale
