"""
Report Generator Agent — Compiles statistical results into a structured report.

Uses LLM for narrative generation (with template fallback),
deterministic data for tables/stats, and chart_generator for figures.
"""
from typing import Any, List, Dict, Optional

from src.agentic_core.agents.base import BaseAgent
from src.agentic_core.agents.state import MissionState
from src.agentic_core.agents.context import AgentContext, AgentPlan, AgentAction
from src.medstats.agents.report_generator.prompts import REPORT_SYSTEM_PROMPT, NARRATIVE_USER_TEMPLATE
from src.medstats.agents.report_generator.formatters import (
    group_top_categories,
    generate_methodology_note,
    generate_narrative_template,
)
from src.medstats.core.chart_generator import select_and_generate_chart
from src.medstats.agents.rag_knowledge import get_rag_agent
from src.shared.logging.logger import get_logger

logger = get_logger(__name__)


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

        opening = (
            f"A {'prospective' if data['study_design'] == 'single_cohort' else 'comparative'} study "
            f"was conducted to evaluate {title}. "
            f"{sample_size} subjects fulfilling the selection criteria were recruited for the study."
        )
        sections.append({"type": "heading", "content": "OBSERVATIONS AND RESULTS"})
        sections.append({"type": "paragraph", "content": opening})

        for result in test_results:
            if "variable_name" not in result or "descriptive" not in result:
                continue
            if result.get("descriptive") is None:
                continue

            table_number += 1
            var_name = result["variable_name"]
            descriptive = result["descriptive"]
            freq_table = result.get("frequency_table")
            test_data = result.get("test_result")
            if test_data is not None and not isinstance(test_data, dict):
                test_data = None

            sections.append({
                "type": "table_heading",
                "content": f"Table No.{table_number}: {var_name.upper().replace('_', ' ')}",
            })

            narrative = await self._generate_narrative(
                var_name, descriptive, freq_table, test_data, sample_size,
                study_design=data.get("study_design", ""),
            )
            sections.append({"type": "summary_text", "content": narrative})

            table_section = self._build_intelligent_table(
                var_name, descriptive, freq_table, test_data, sample_size
            )
            if table_section:
                sections.append(table_section)

            if descriptive.get("mean") is not None:
                sections.append({
                    "type": "descriptive_note",
                    "content": f"Mean \u00b1 SD: {descriptive['mean']} \u00b1 {descriptive['std']}",
                })

            if test_data and test_data.get("significance_statement"):
                sections.append({"type": "test_result", "content": test_data["significance_statement"]})

            data_type = self._infer_data_type(descriptive, freq_table)
            raw_values = result.get("raw_values")

            chart_path = select_and_generate_chart(
                variable_name=var_name,
                freq_table=freq_table,
                descriptive=descriptive,
                data_type=data_type,
                raw_values=raw_values,
                sample_size=sample_size,
            )
            if chart_path:
                charts.append({"variable": var_name, "path": chart_path, "table_number": table_number})
                sections.append({
                    "type": "chart",
                    "variable_name": var_name,
                    "chart_path": chart_path,
                })

        tests_used = set()
        for r in test_results:
            td = r.get("test_result")
            if td and isinstance(td, dict) and td.get("test_type"):
                tests_used.add(td["test_type"])

        methodology = generate_methodology_note(tests_used, sample_size)
        sections.append({"type": "heading", "content": "Statistical Analysis"})
        sections.append({"type": "paragraph", "content": methodology})

        summary = await self._generate_summary(test_results, title, sample_size)

        state.agent_outputs["report_sections"] = sections
        state.agent_outputs["charts"] = charts
        state.agent_outputs["summary"] = summary
        state.agent_outputs["methodology_note"] = methodology

        action = AgentAction.success_action(
            "report_generation",
            output=f"Generated report with {table_number} tables, {len(charts)} charts",
        )
        return [action], state

    # ------------------------------------------------------------------
    # Intelligent table builder
    # ------------------------------------------------------------------

    def _build_intelligent_table(
        self, var_name: str, descriptive: dict, freq_table, test_data, sample_size: int
    ) -> Optional[dict]:
        """Build a context-appropriate table format based on data type."""
        if not freq_table or not isinstance(freq_table, dict) or not freq_table.get("rows"):
            if descriptive and descriptive.get("mean") is not None:
                return self._build_stats_summary_table(var_name, descriptive, sample_size)
            return None

        rows = [r for r in freq_table["rows"] if r["category"] != "TOTAL"]
        total_row = next((r for r in freq_table["rows"] if r["category"] == "TOTAL"), None)

        if not rows:
            return None

        if len(rows) > 12:
            rows = group_top_categories(rows, top_n=10)

        is_continuous = descriptive and descriptive.get("mean") is not None
        has_test = test_data and isinstance(test_data, dict) and test_data.get("p_value") is not None

        if is_continuous and has_test:
            return self._build_comprehensive_table(var_name, descriptive, rows, test_data, sample_size)
        elif is_continuous:
            return self._build_range_distribution_table(var_name, rows, descriptive, sample_size)
        elif len(rows) == 2:
            return self._build_binary_table(var_name, rows, sample_size)
        else:
            return self._build_frequency_table(var_name, rows, total_row, sample_size)

    def _build_frequency_table(self, var_name: str, rows: list, total_row, sample_size: int) -> dict:
        """Standard frequency table with number and percentage."""
        display_name = var_name.replace("_", " ").title()
        headers = [display_name, f"Number (n={sample_size})", "Percentage"]
        table_rows = []

        for r in rows:
            table_rows.append([r["category"], str(r["count"]), f"{r['percentage']:.2f}%"])

        if total_row:
            table_rows.append(["TOTAL", str(total_row["count"]), "100.00%"])

        return {
            "type": "table",
            "table_format": "frequency",
            "variable_name": var_name,
            "headers": headers,
            "rows": table_rows,
        }

    def _build_binary_table(self, var_name: str, rows: list, sample_size: int) -> dict:
        """Compact two-row table for binary variables."""
        display_name = var_name.replace("_", " ").title()
        headers = [display_name, "Number", "Percentage", "Ratio"]

        total = sum(r["count"] for r in rows)
        table_rows = []
        for r in rows:
            ratio = f"{r['count']}:{total - r['count']}" if total > 0 else "-"
            table_rows.append([r["category"], str(r["count"]), f"{r['percentage']:.2f}%", ratio])

        table_rows.append(["TOTAL", str(total), "100.00%", "-"])

        return {
            "type": "table",
            "table_format": "binary",
            "variable_name": var_name,
            "headers": headers,
            "rows": table_rows,
        }

    def _build_range_distribution_table(
        self, var_name: str, rows: list, descriptive: dict, sample_size: int
    ) -> dict:
        """Table for continuous data with range groupings and cumulative percentages."""
        display_name = var_name.replace("_", " ").title()
        headers = [display_name, "Number", "Percentage", "Cumulative %"]
        table_rows = []
        cumulative = 0.0

        for r in rows:
            cumulative += r["percentage"]
            table_rows.append([
                r["category"], str(r["count"]),
                f"{r['percentage']:.2f}%", f"{cumulative:.2f}%"
            ])

        total = sum(r["count"] for r in rows)
        table_rows.append(["TOTAL", str(total), "100.00%", "—"])
        table_rows.append([
            f"Mean ± SD",
            f"{descriptive['mean']:.2f} ± {descriptive.get('std', 0):.2f}",
            f"Range: {descriptive.get('min_val', '—')} – {descriptive.get('max_val', '—')}",
            ""
        ])

        return {
            "type": "table",
            "table_format": "range_distribution",
            "variable_name": var_name,
            "headers": headers,
            "rows": table_rows,
        }

    def _build_comprehensive_table(
        self, var_name: str, descriptive: dict, rows: list, test_data: dict, sample_size: int
    ) -> dict:
        """Rich table including frequency, stats, and test result."""
        display_name = var_name.replace("_", " ").title()
        headers = [display_name, f"Number (n={sample_size})", "Percentage", "Statistical Test"]

        table_rows = []
        for i, r in enumerate(rows):
            test_col = ""
            if i == 0:
                test_name = test_data.get("test_type", "").replace("_", " ").title()
                p_val = test_data.get("p_value")
                if p_val is not None:
                    test_col = f"{test_name}\np={p_val:.4f}"
            table_rows.append([r["category"], str(r["count"]), f"{r['percentage']:.2f}%", test_col])

        total = sum(r["count"] for r in rows)
        sig = "Significant" if test_data.get("is_significant_05") else "Not significant"
        table_rows.append(["TOTAL", str(total), "100.00%", f"({sig} at p<0.05)"])

        return {
            "type": "table",
            "table_format": "comprehensive",
            "variable_name": var_name,
            "headers": headers,
            "rows": table_rows,
        }

    def _build_stats_summary_table(self, var_name: str, descriptive: dict, sample_size: int) -> dict:
        """Descriptive statistics summary table for continuous variables without frequency bins."""
        display_name = var_name.replace("_", " ").title()
        headers = ["Statistic", "Value"]
        table_rows = [
            ["N", str(sample_size)],
            ["Mean", f"{descriptive.get('mean', 0):.2f}"],
            ["Standard Deviation", f"{descriptive.get('std', 0):.2f}"],
            ["Minimum", str(descriptive.get("min_val", "—"))],
            ["Maximum", str(descriptive.get("max_val", "—"))],
            ["Median", str(descriptive.get("median", "—"))],
        ]

        return {
            "type": "table",
            "table_format": "stats_summary",
            "variable_name": var_name,
            "headers": headers,
            "rows": table_rows,
        }

    # ------------------------------------------------------------------
    # Data type inference
    # ------------------------------------------------------------------

    def _infer_data_type(self, descriptive: dict, freq_table) -> str:
        """Infer whether a variable is continuous or categorical."""
        if descriptive and descriptive.get("mean") is not None:
            return "continuous"
        return "categorical"

    # ------------------------------------------------------------------
    # Narrative generation
    # ------------------------------------------------------------------

    async def _generate_narrative(
        self,
        var_name: str,
        descriptive: dict,
        freq_table: dict,
        test_data,
        sample_size: int,
        study_design: str = "",
    ) -> str:
        """Generate narrative paragraph for a variable using LLM or template fallback."""
        if not self.services:
            return generate_narrative_template(var_name, descriptive, freq_table, sample_size)

        prompt = NARRATIVE_USER_TEMPLATE.format(
            var_name=var_name,
            sample_size=sample_size,
            descriptive=descriptive,
            freq_table=freq_table,
            test_data=test_data,
        )

        rag_agent = get_rag_agent()
        if rag_agent and test_data and isinstance(test_data, dict):
            test_type = test_data.get("test_type", "")
            tags = [t for t in [test_type, study_design] if t]
            query = (
                f"clinical interpretation of {test_type} results for {var_name} "
                f"in a {study_design or 'medical'} study with N={sample_size}"
            )
            rag_result = await rag_agent.retrieve(query, tags=tags)
            rag_block = rag_agent.format_context_block(rag_result)
            if rag_block:
                prompt = f"{rag_block}\n\n{prompt}"

        try:
            narrative = await self.services.generate_text(
                prompt=prompt,
                system_prompt=REPORT_SYSTEM_PROMPT,
                temperature=0.3,
            )
            return narrative.strip()
        except Exception as e:
            logger.warning("LLM narrative generation failed, using template", error=str(e))
            return generate_narrative_template(var_name, descriptive, freq_table, sample_size)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    async def _generate_summary(self, test_results: list, title: str, sample_size: int) -> str:
        """Generate a structured executive summary with study description, methodology, and key findings."""
        tests_used = {}
        significant = []
        non_significant = []

        for r in test_results:
            td = r.get("test_result")
            if td and isinstance(td, dict):
                test_type = td.get("test_type", "")
                if test_type:
                    tests_used[test_type] = tests_used.get(test_type, 0) + 1
                if td.get("is_significant_05"):
                    significant.append(r)
                else:
                    non_significant.append(r)

        study_desc = self._build_study_description(title, sample_size, test_results)
        methodology_desc = self._build_methodology_description(tests_used, sample_size)
        findings_list = self._build_findings_list(significant, non_significant)

        if self.services:
            rag_agent = get_rag_agent()
            rag_block = ""
            if rag_agent and tests_used:
                test_list = ", ".join(tests_used.keys())
                rag_result = await rag_agent.retrieve(
                    f"methodology rationale for statistical tests: {test_list} in medical thesis N={sample_size}",
                    tags=list(tests_used.keys())[:5],
                )
                rag_block = rag_agent.format_context_block(rag_result)

            prompt = (
                f"Write a professional executive summary for a medical thesis statistical analysis.\n\n"
                f"Study: {title}\n"
                f"Sample size: N={sample_size}\n"
                f"Total variables: {len(test_results)}\n"
                f"Tests used: {', '.join(tests_used.keys())}\n\n"
                f"Significant findings ({len(significant)}):\n"
                + "\n".join(f"- {s['variable_name']}: {s['test_result'].get('significance_statement','')}" for s in significant[:8])
                + f"\n\nNon-significant ({len(non_significant)}):\n"
                + "\n".join(f"- {r['variable_name']}" for r in non_significant[:5])
                + "\n\nWrite ONLY the study introduction paragraph (2-3 sentences) describing what this study is about. "
                "Use formal academic medical language. Do not include methodology or findings — just the study description."
            )
            if rag_block:
                prompt = f"{rag_block}\n\n{prompt}"
            try:
                llm_intro = await self.services.generate_text(
                    prompt=prompt,
                    system_prompt=REPORT_SYSTEM_PROMPT,
                    temperature=0.3,
                )
                if llm_intro and llm_intro.strip():
                    study_desc = llm_intro.strip()
            except Exception as e:
                logger.warning("LLM summary generation failed", error=str(e))

        import json
        return json.dumps({
            "study_description": study_desc,
            "methodology": methodology_desc,
            "key_findings": findings_list,
            "stats": {
                "total_variables": len(test_results),
                "significant": len(significant),
                "not_significant": len(non_significant),
            }
        })

    def _build_study_description(self, title: str, sample_size: int, test_results: list) -> str:
        """Build a template study description paragraph."""
        seen = set()
        variables_list = []
        for r in test_results:
            vname = r.get("variable_name", "").replace("_", " ").title()
            if vname and vname.lower() not in seen:
                seen.add(vname.lower())
                variables_list.append(vname)
        vars_text = ", ".join(variables_list[:4])
        return (
            f"A prospective study was conducted to evaluate {title.lower()}. "
            f"{sample_size} subjects fulfilling the selection criteria were recruited for the study. "
            f"The key variables analyzed include {vars_text}, among others."
        )

    def _build_methodology_description(self, tests_used: dict, sample_size: int) -> str:
        """Build methodology paragraph explaining which tests were used and why."""
        from src.medstats.agents.report_generator.schemas import TEST_NAME_MAP, TEST_RATIONALE_MAP

        parts = []
        parts.append(
            f"Statistical analysis was performed on a sample of {sample_size} subjects. "
            f"Data was analyzed using descriptive statistics (mean, standard deviation, percentages) "
            f"and inferential statistical tests."
        )

        if tests_used:
            test_explanations = []
            for test_key, count in tests_used.items():
                full_name = TEST_NAME_MAP.get(test_key, test_key.replace("_", " ").title())
                rationale = TEST_RATIONALE_MAP.get(test_key, "to assess statistical significance")
                test_explanations.append(f"{full_name} (applied to {count} variable{'s' if count > 1 else ''}, {rationale})")

            parts.append("The following tests were employed: " + "; ".join(test_explanations) + ".")
            parts.append("A p-value of less than 0.05 was considered statistically significant.")

        return " ".join(parts)

    def _build_findings_list(self, significant: list, non_significant: list) -> list:
        """Build structured key findings list."""
        findings = []

        for s in significant:
            td = s.get("test_result", {})
            var_name = s["variable_name"].replace("_", " ").title()
            p_val = td.get("p_value")
            stmt = td.get("significance_statement", "statistically significant")
            findings.append({
                "variable": var_name,
                "result": stmt,
                "p_value": p_val,
                "significant": True,
            })

        for ns in non_significant[:5]:
            td = ns.get("test_result", {})
            var_name = ns["variable_name"].replace("_", " ").title()
            p_val = td.get("p_value")
            findings.append({
                "variable": var_name,
                "result": f"p={p_val:.4f}, not significant" if p_val else "Not significant",
                "p_value": p_val,
                "significant": False,
            })

        return findings
