"""
Report Generator Agent - Compiles statistical results into a structured report.
Uses LLM for narrative generation, deterministic data for tables/stats.
"""
from typing import Any, List, Dict
from src.agentic_core.agents.base_agent import BaseAgent, AgentContext, AgentPlan, AgentAction, MissionState
from src.medstats.core.chart_generator import generate_pie_chart, generate_bar_chart
from src.shared.logging.logger import get_logger

logger = get_logger(__name__)

REPORT_SYSTEM_PROMPT = """You are a medical biostatistician writing the "Observations and Results" 
chapter of a medical thesis. Write in formal academic English. Be concise and factual.
Do NOT invent any numbers - use ONLY the statistics provided to you.
Follow this pattern for each variable:
1. State the finding narratively (e.g., "Majority of patients (X%) were in Y category")
2. Reference the table number
3. If a test was applied, state the result with p-value"""


class ReportGeneratorAgent(BaseAgent):
    role_name = "ReportGenerator"
    requires_llm_reasoning = False

    async def _perceive_domain(self, state: MissionState, context: AgentContext) -> AgentContext:
        """Gather all test results and metadata."""
        context.parsed_data = {
            "test_results": state.agent_outputs.get("test_results", []),
            "frequency_tables": state.agent_outputs.get("frequency_tables", []),
            "title": state.agent_outputs.get("title", ""),
            "intent": state.agent_outputs.get("intent", ""),
            "sample_size": state.agent_outputs.get("sample_size", 0),
            "study_design": state.agent_outputs.get("study_design", ""),
            "variables": state.agent_outputs.get("variables", []),
        }
        return context

    async def _act_domain(self, state: MissionState, context: AgentContext, plan: AgentPlan):
        """Generate report sections with narrative, tables, and charts."""
        data = context.parsed_data
        test_results = data["test_results"]
        sample_size = data["sample_size"]
        title = data["title"]

        sections = []
        charts = []
        table_number = 0

        # Opening paragraph
        opening = (
            f"A {'prospective' if data['study_design'] == 'single_cohort' else 'comparative'} study "
            f"was conducted to evaluate {title}. "
            f"{sample_size} subjects fulfilling the selection criteria were recruited for the study."
        )
        sections.append({
            "type": "heading",
            "content": "OBSERVATIONS AND RESULTS",
        })
        sections.append({
            "type": "paragraph",
            "content": opening,
        })

        # Generate per-variable sections
        for result in test_results:
            if "variable_name" not in result or "descriptive" not in result:
                continue

            table_number += 1
            var_name = result["variable_name"]
            descriptive = result["descriptive"]
            freq_table = result.get("frequency_table")
            test_data = result.get("test_result")

            # Section heading
            sections.append({
                "type": "table_heading",
                "content": f"Table No.{table_number}: {var_name.upper().replace('_', ' ')}",
            })

            # Generate narrative for this variable
            narrative = await self._generate_narrative(
                var_name, descriptive, freq_table, test_data, sample_size
            )
            sections.append({
                "type": "paragraph",
                "content": narrative,
            })

            # Frequency table
            if freq_table and freq_table.get("rows"):
                sections.append({
                    "type": "table",
                    "variable_name": var_name,
                    "headers": ["Category", "Number", "Percentage"],
                    "rows": [
                        [r["category"], str(r["count"]), f"{r['percentage']:.2f}%"]
                        for r in freq_table["rows"]
                    ],
                })

                # Mean ± SD row (if numerical)
                if descriptive.get("mean") is not None:
                    sections.append({
                        "type": "descriptive_note",
                        "content": f"Mean ± SD: {descriptive['mean']} ± {descriptive['std']}",
                    })

            # Statistical test result
            if test_data and test_data.get("significance_statement"):
                sections.append({
                    "type": "test_result",
                    "content": test_data["significance_statement"],
                })

            # Generate chart
            chart_path = self._generate_chart_for_variable(var_name, freq_table, descriptive)
            if chart_path:
                charts.append({"variable": var_name, "path": chart_path, "table_number": table_number})
                sections.append({
                    "type": "chart",
                    "variable_name": var_name,
                    "chart_path": chart_path,
                })

        # Methodology note
        tests_used = set()
        for r in test_results:
            if r.get("test_result") and r["test_result"].get("test_type"):
                tests_used.add(r["test_result"]["test_type"])

        methodology = self._generate_methodology_note(tests_used, sample_size)
        sections.append({"type": "heading", "content": "Statistical Analysis"})
        sections.append({"type": "paragraph", "content": methodology})

        # Summary
        summary = await self._generate_summary(test_results, title, sample_size)

        # Store in state
        state.agent_outputs["report_sections"] = sections
        state.agent_outputs["charts"] = charts
        state.agent_outputs["summary"] = summary
        state.agent_outputs["methodology_note"] = methodology

        action = AgentAction.success_action(
            "report_generation",
            output=f"Generated report with {table_number} tables, {len(charts)} charts",
        )
        return [action], state

    async def _generate_narrative(
        self, var_name: str, descriptive: dict, freq_table: dict, test_data: dict, sample_size: int
    ) -> str:
        """Generate narrative paragraph for a variable using LLM."""
        if not self.services:
            return self._generate_narrative_template(var_name, descriptive, freq_table, sample_size)

        prompt = f"""Write a 1-2 sentence narrative for the following variable in a medical thesis results section:

Variable: {var_name}
Sample size: {sample_size}
Descriptive stats: {descriptive}
Frequency table: {freq_table}
Test result: {test_data}

Write ONLY the narrative paragraph. Use the exact numbers provided. Do not add any test result statements."""

        try:
            narrative = await self.services.generate_text(
                prompt=prompt,
                system_prompt=REPORT_SYSTEM_PROMPT,
                temperature=0.3,
            )
            return narrative.strip()
        except Exception as e:
            logger.warning("LLM narrative generation failed, using template", error=str(e))
            return self._generate_narrative_template(var_name, descriptive, freq_table, sample_size)

    def _generate_narrative_template(
        self, var_name: str, descriptive: dict, freq_table: dict, sample_size: int
    ) -> str:
        """Fallback template-based narrative."""
        if descriptive.get("mean") is not None:
            return (
                f"The {var_name.lower()} ranged from {descriptive.get('min_val')} to "
                f"{descriptive.get('max_val')} with a mean of {descriptive['mean']} "
                f"and standard deviation of {descriptive['std']}."
            )
        elif freq_table and freq_table.get("rows"):
            rows = [r for r in freq_table["rows"] if r["category"] != "TOTAL"]
            if rows:
                top = rows[0]
                return (
                    f"Among {sample_size} subjects, the majority ({top['percentage']:.1f}%) "
                    f"had {top['category'].lower()} as the {var_name.lower()}."
                )
        return f"The distribution of {var_name.lower()} is presented in the table below."

    def _generate_chart_for_variable(
        self, var_name: str, freq_table: dict, descriptive: dict
    ) -> str:
        """Generate appropriate chart based on variable type."""
        if not freq_table or not freq_table.get("rows"):
            return ""

        rows = [r for r in freq_table["rows"] if r["category"] != "TOTAL"]
        if not rows:
            return ""

        labels = [r["category"] for r in rows]
        values = [r["count"] for r in rows]

        safe_name = var_name.replace(" ", "_").replace("/", "_").lower()

        if len(labels) <= 6:
            return generate_pie_chart(
                labels=labels,
                values=values,
                title=f"Distribution of {var_name}",
                variable_name=safe_name,
            )
        else:
            return generate_bar_chart(
                categories=labels,
                values=values,
                title=f"Distribution of {var_name}",
                variable_name=safe_name,
                ylabel="Number of patients",
            )

    def _generate_methodology_note(self, tests_used: set, sample_size: int) -> str:
        """Generate the statistical analysis methodology paragraph."""
        test_names = {
            "z_test_proportion": "Z-test of proportions",
            "chi_square": "Chi-square test",
            "fisher_exact": "Fisher's exact test",
            "unpaired_t_test": "Student's unpaired t-test",
            "paired_t_test": "Student's paired t-test",
            "mann_whitney_u": "Mann-Whitney U test",
            "one_way_anova": "One-way ANOVA",
            "kruskal_wallis": "Kruskal-Wallis test",
            "shapiro_wilk": "Shapiro-Wilk test",
        }

        named_tests = [test_names.get(t, t) for t in tests_used]
        tests_str = ", ".join(named_tests) if named_tests else "descriptive statistics"

        return (
            f"Statistical analysis was done by descriptive statistics as mean, SD, "
            f"percentage/proportions. Comparison of variables was done by applying "
            f"{tests_str} at 5% (p<0.05) and 1% (p<0.01) level of significance. "
            f"Statistical analysis was performed using MedStats AI platform with "
            f"scipy and statsmodels libraries for Python."
        )

    async def _generate_summary(self, test_results: list, title: str, sample_size: int) -> str:
        """Generate executive summary of findings."""
        significant = [
            r for r in test_results
            if r.get("test_result") and r["test_result"].get("is_significant_05")
        ]

        summary_parts = [f"Study: {title}", f"Sample size: {sample_size}"]
        if significant:
            summary_parts.append(f"Significant findings: {len(significant)}")
            for s in significant[:5]:
                summary_parts.append(f"- {s['variable_name']}: {s['test_result']['significance_statement']}")

        return "\n".join(summary_parts)
