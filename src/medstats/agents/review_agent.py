"""
AI Professor Review Agent - Automated quality review of generated analysis reports.

Three-tier validation:
  Tier 1: Mechanical/Clerical (deterministic, no LLM)
  Tier 2: Statistical Methodology (rule-based + heuristics)
  Tier 3: Clinical Interpretation (LLM-powered, flags for professor)

Produces a structured review summary that dramatically reduces professor review time.
"""
from typing import List, Dict, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum
import pandas as pd
from io import StringIO

from src.agentic_core.agents.base_agent import BaseAgent, AgentContext, AgentPlan, AgentAction, MissionState
from src.medstats.core.stats_engine import StatsEngine
from src.shared.logging.logger import get_logger

logger = get_logger(__name__)


class Severity(str, Enum):
    OK = "ok"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class ReviewItem:
    check_id: str
    tier: int
    severity: Severity
    category: str
    message: str
    variable: str = ""
    suggestion: str = ""
    auto_fixable: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ReviewSummary:
    total_checks: int = 0
    passed: int = 0
    info_count: int = 0
    warnings: int = 0
    errors: int = 0
    estimated_review_minutes: int = 0
    items: List[ReviewItem] = field(default_factory=list)
    overall_verdict: str = "pending"

    def to_dict(self) -> dict:
        return {
            "total_checks": self.total_checks,
            "passed": self.passed,
            "info_count": self.info_count,
            "warnings": self.warnings,
            "errors": self.errors,
            "estimated_review_minutes": self.estimated_review_minutes,
            "overall_verdict": self.overall_verdict,
            "items": [i.to_dict() for i in self.items if i.severity != Severity.OK],
            "all_items": [i.to_dict() for i in self.items],
        }


class ReviewAgent(BaseAgent):
    """
    AI Professor Review Agent.
    Runs after ReportGeneratorAgent to validate the generated report.
    """
    role_name = "ReviewAgent"
    requires_llm_reasoning = False

    def __init__(self, services=None):
        super().__init__(services=services)
        self.engine = StatsEngine()

    @staticmethod
    def _get_inner(tr: dict) -> dict:
        """Extract the inner test result, handling both nested and flat formats."""
        inner = tr.get("test_result")
        if inner is None or not isinstance(inner, dict):
            return tr
        return inner

    async def _perceive_domain(self, state: MissionState, context: AgentContext) -> AgentContext:
        """Gather report data for review."""
        # Get dataframe JSON - handle both formats (direct JSON or ParsedDataset object)
        df_json = state.agent_outputs.get("dataframe_json", "")
        if not df_json:
            parsed = state.agent_outputs.get("parsed_dataset")
            if parsed and hasattr(parsed, "dataframe"):
                df_json = parsed.dataframe.to_json()
            elif parsed and isinstance(parsed, str):
                df_json = parsed

        context.parsed_data = {
            "report_sections": state.agent_outputs.get("report_sections", []),
            "test_results": state.agent_outputs.get("test_results", []),
            "frequency_tables": state.agent_outputs.get("frequency_tables", []),
            "variables": state.agent_outputs.get("variables", []),
            "sample_size": state.agent_outputs.get("sample_size", 0),
            "study_design": state.agent_outputs.get("study_design", ""),
            "charts": state.agent_outputs.get("charts", []),
            "title": state.agent_outputs.get("title", ""),
            "dataframe_json": df_json,
        }
        return context

    async def _act_domain(self, state: MissionState, context: AgentContext, plan: AgentPlan):
        """Run all three tiers of review."""
        data = context.parsed_data
        items: List[ReviewItem] = []

        # Load dataframe if available
        df = None
        if data["dataframe_json"]:
            try:
                df = pd.read_json(StringIO(data["dataframe_json"]))
            except Exception:
                pass

        # Tier 1: Mechanical/Clerical checks
        items.extend(self._tier1_mechanical(data, df))

        # Tier 2: Statistical Methodology checks
        items.extend(self._tier2_methodology(data, df))

        # Tier 3: Clinical Interpretation (LLM if available)
        tier3_items = await self._tier3_interpretation(data, df)
        items.extend(tier3_items)

        # Deduplicate: same check_id + variable should appear only once
        seen = set()
        deduped = []
        for item in items:
            key = (item.check_id, item.variable)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        items = deduped

        # Build summary
        summary = ReviewSummary(
            total_checks=len(items),
            passed=sum(1 for i in items if i.severity == Severity.OK),
            info_count=sum(1 for i in items if i.severity == Severity.INFO),
            warnings=sum(1 for i in items if i.severity == Severity.WARNING),
            errors=sum(1 for i in items if i.severity == Severity.ERROR),
            items=items,
        )

        # Estimate review time
        if summary.errors == 0 and summary.warnings <= 2:
            summary.estimated_review_minutes = 5
            summary.overall_verdict = "approve_recommended"
        elif summary.errors == 0 and summary.warnings <= 5:
            summary.estimated_review_minutes = 10
            summary.overall_verdict = "minor_issues"
        elif summary.errors <= 2:
            summary.estimated_review_minutes = 20
            summary.overall_verdict = "needs_attention"
        else:
            summary.estimated_review_minutes = 30
            summary.overall_verdict = "significant_issues"

        state.agent_outputs["review_summary"] = summary.to_dict()
        state.agent_outputs["review_verdict"] = summary.overall_verdict

        action = AgentAction.success_action(
            "review",
            output=(
                f"Review complete: {summary.passed} passed, "
                f"{summary.warnings} warnings, {summary.errors} errors. "
                f"Verdict: {summary.overall_verdict}"
            ),
        )
        return [action], state

    # ─── TIER 1: MECHANICAL/CLERICAL ───────────────────────────────────────────

    def _tier1_mechanical(self, data: dict, df: Optional[pd.DataFrame]) -> List[ReviewItem]:
        """Deterministic checks — no LLM needed."""
        items = []

        # 1.1 Check all input variables appear in report
        items.extend(self._check_variable_coverage(data))

        # 1.2 Check percentages sum to 100%
        items.extend(self._check_percentage_sums(data))

        # 1.3 Check table numbering is sequential
        items.extend(self._check_table_numbering(data))

        # 1.4 Check narrative numbers match table data
        items.extend(self._check_narrative_consistency(data))

        # 1.5 Check sample size consistency
        items.extend(self._check_sample_size_consistency(data, df))

        # 1.6 Check charts generated for each variable
        items.extend(self._check_chart_coverage(data))

        return items

    def _check_variable_coverage(self, data: dict) -> List[ReviewItem]:
        """Ensure all input variables have corresponding report sections."""
        items = []
        variables = data.get("variables", [])
        sections = data.get("report_sections", [])
        test_results = data.get("test_results", [])

        reported_vars = set()
        for s in sections:
            if s.get("variable_name"):
                reported_vars.add(s["variable_name"])
        for tr in test_results:
            if tr.get("variable_name"):
                reported_vars.add(tr["variable_name"])

        # Exclude identifier columns
        skip_patterns = {"NO", "IPD", "NAME", "DATE", "DIAGNOSIS", "S.NO", "SR.NO"}

        for var in variables:
            name = var["name"]
            if name.upper() in skip_patterns:
                continue
            if name in reported_vars:
                items.append(ReviewItem(
                    check_id="var_coverage",
                    tier=1,
                    severity=Severity.OK,
                    category="Variable Coverage",
                    message=f"{name} is included in report",
                    variable=name,
                ))
            else:
                items.append(ReviewItem(
                    check_id="var_missing",
                    tier=1,
                    severity=Severity.ERROR,
                    category="Variable Coverage",
                    message=f"Variable '{name}' from input data is NOT in the report",
                    variable=name,
                    suggestion="Add analysis for this variable or document why it was excluded",
                    auto_fixable=True,
                ))
        return items

    def _check_percentage_sums(self, data: dict) -> List[ReviewItem]:
        """Verify frequency table percentages sum to ~100%."""
        items = []
        freq_tables = data.get("frequency_tables", [])

        for ft in freq_tables:
            var_name = ft.get("variable_name", "unknown")
            rows = ft.get("rows", [])
            data_rows = [r for r in rows if r.get("category") != "TOTAL"]

            if not data_rows:
                continue

            total_pct = sum(r.get("percentage", 0) for r in data_rows)
            if abs(total_pct - 100.0) < 1.0:
                items.append(ReviewItem(
                    check_id="pct_sum",
                    tier=1,
                    severity=Severity.OK,
                    category="Percentage Validation",
                    message=f"{var_name}: percentages sum to {total_pct:.1f}%",
                    variable=var_name,
                ))
            else:
                items.append(ReviewItem(
                    check_id="pct_sum_error",
                    tier=1,
                    severity=Severity.WARNING,
                    category="Percentage Validation",
                    message=f"{var_name}: percentages sum to {total_pct:.1f}% (expected 100%)",
                    variable=var_name,
                    suggestion="Check for rounding errors or missing categories",
                    auto_fixable=True,
                ))
        return items

    def _check_table_numbering(self, data: dict) -> List[ReviewItem]:
        """Verify table numbers are sequential."""
        sections = data.get("report_sections", [])
        table_numbers = []

        for s in sections:
            if s.get("type") == "table_heading":
                content = s.get("content", "")
                import re
                match = re.search(r"Table\s*No\.?\s*(\d+)", content, re.IGNORECASE)
                if match:
                    table_numbers.append(int(match.group(1)))

        if not table_numbers:
            return []

        items = []
        expected = list(range(1, max(table_numbers) + 1))
        if table_numbers == expected:
            items.append(ReviewItem(
                check_id="table_seq",
                tier=1,
                severity=Severity.OK,
                category="Table Numbering",
                message=f"Tables numbered sequentially 1-{len(table_numbers)}",
            ))
        else:
            missing = set(expected) - set(table_numbers)
            items.append(ReviewItem(
                check_id="table_seq_error",
                tier=1,
                severity=Severity.WARNING,
                category="Table Numbering",
                message=f"Table numbering gaps: missing {missing}",
                suggestion="Renumber tables sequentially",
                auto_fixable=True,
            ))
        return items

    def _check_narrative_consistency(self, data: dict) -> List[ReviewItem]:
        """Check that narrative paragraphs reference correct numbers."""
        items = []
        sections = data.get("report_sections", [])
        sample_size = data.get("sample_size", 0)

        for s in sections:
            if s.get("type") == "paragraph":
                content = s.get("content", "")
                # Check if sample size mentioned matches
                import re
                size_mentions = re.findall(r"(\d+)\s*subjects?", content)
                for mentioned in size_mentions:
                    if int(mentioned) != sample_size and int(mentioned) != sample_size - 1:
                        items.append(ReviewItem(
                            check_id="sample_mismatch",
                            tier=1,
                            severity=Severity.WARNING,
                            category="Data Consistency",
                            message=f"Narrative mentions {mentioned} subjects but sample size is {sample_size}",
                            suggestion=f"Update to {sample_size} subjects",
                            auto_fixable=True,
                        ))

        if not any(i.check_id == "sample_mismatch" for i in items):
            items.append(ReviewItem(
                check_id="narrative_ok",
                tier=1,
                severity=Severity.OK,
                category="Data Consistency",
                message="Narrative numbers consistent with data",
            ))
        return items

    def _check_sample_size_consistency(self, data: dict, df: Optional[pd.DataFrame]) -> List[ReviewItem]:
        """Check sample size is consistent across sections."""
        items = []
        reported_size = data.get("sample_size", 0)

        if df is not None:
            # Drop fully-null rows
            actual_size = len(df.dropna(how="all"))
            if abs(actual_size - reported_size) > 2:
                items.append(ReviewItem(
                    check_id="sample_size_drift",
                    tier=1,
                    severity=Severity.WARNING,
                    category="Sample Size",
                    message=f"Reported N={reported_size} but data has {actual_size} non-null rows",
                    suggestion="Clarify inclusion/exclusion criteria",
                ))
            else:
                items.append(ReviewItem(
                    check_id="sample_size_ok",
                    tier=1,
                    severity=Severity.OK,
                    category="Sample Size",
                    message=f"Sample size N={reported_size} confirmed",
                ))
        return items

    def _check_chart_coverage(self, data: dict) -> List[ReviewItem]:
        """Verify charts exist for reported variables."""
        items = []
        charts = data.get("charts", [])
        test_results = data.get("test_results", [])

        charted_vars = {c.get("variable", "") for c in charts}
        analyzed_vars = {tr.get("variable_name", "") for tr in test_results if tr.get("variable_name")}

        missing_charts = analyzed_vars - charted_vars - {""}
        if missing_charts:
            items.append(ReviewItem(
                check_id="missing_charts",
                tier=1,
                severity=Severity.INFO,
                category="Visualisation",
                message=f"{len(missing_charts)} variable(s) analyzed but no chart: {', '.join(list(missing_charts)[:3])}",
                suggestion="Consider adding charts for better visual representation",
            ))
        else:
            items.append(ReviewItem(
                check_id="charts_ok",
                tier=1,
                severity=Severity.OK,
                category="Visualisation",
                message=f"All {len(charts)} charts present for analyzed variables",
            ))
        return items

    # ─── TIER 2: STATISTICAL METHODOLOGY ──────────────────────────────────────

    def _tier2_methodology(self, data: dict, df: Optional[pd.DataFrame]) -> List[ReviewItem]:
        """Rule-based statistical methodology checks."""
        items = []

        # 2.1 Test-data type appropriateness (basic type match)
        items.extend(self._check_test_appropriateness(data))

        # 2.2 Deep test selection validation (normality, groups, study design)
        items.extend(self._check_test_selection_logic(data, df))

        # 2.3 Sample size adequacy per test
        items.extend(self._check_sample_adequacy(data, df))

        # 2.4 Normality assumption validation
        items.extend(self._check_normality_assumptions(data, df))

        # 2.5 Multiple testing correction
        items.extend(self._check_multiple_testing(data))

        # 2.6 Chi-square expected cell counts
        items.extend(self._check_chi_square_assumptions(data, df))

        return items

    def _check_test_appropriateness(self, data: dict) -> List[ReviewItem]:
        """Verify the right test was used for the data type."""
        items = []
        variables = {v["name"]: v for v in data.get("variables", [])}
        test_results = data.get("test_results", [])

        categorical_tests = {"z_test_proportion", "chi_square", "fisher_exact"}
        numerical_tests = {"unpaired_t_test", "paired_t_test", "mann_whitney_u", "one_way_anova", "kruskal_wallis", "shapiro_wilk"}

        for tr in test_results:
            var_name = tr.get("variable_name", "")
            # Handle both nested and flat result formats
            inner = self._get_inner(tr)
            test_type = inner.get("test_type", "") or tr.get("test_type", "")
            if not test_type or not var_name:
                continue

            var_info = variables.get(var_name)
            if not var_info:
                continue

            dtype = var_info.get("dtype", "")
            mismatch = False

            if dtype == "categorical" and test_type in numerical_tests:
                mismatch = True
            elif dtype == "numerical" and test_type in categorical_tests:
                mismatch = True

            if mismatch:
                items.append(ReviewItem(
                    check_id="test_type_mismatch",
                    tier=2,
                    severity=Severity.ERROR,
                    category="Test Appropriateness",
                    message=f"{var_name}: {test_type} used for {dtype} data",
                    variable=var_name,
                    suggestion=f"Use {'chi-square/Fisher' if dtype == 'categorical' else 't-test/Mann-Whitney'} instead",
                    auto_fixable=True,
                ))
            else:
                items.append(ReviewItem(
                    check_id="test_type_ok",
                    tier=2,
                    severity=Severity.OK,
                    category="Test Appropriateness",
                    message=f"{var_name}: {test_type} appropriate for {dtype} data",
                    variable=var_name,
                ))
        return items

    def _check_test_selection_logic(self, data: dict, df: Optional[pd.DataFrame]) -> List[ReviewItem]:
        """
        Deep validation of statistical test selection against the actual data characteristics.
        Checks normality-based decisions, group counts, study design consistency, and
        whether parametric vs non-parametric was correctly chosen.
        """
        items = []
        variables = {v["name"]: v for v in data.get("variables", [])}
        test_results = data.get("test_results", [])
        study_design = data.get("study_design", "")
        sample_size = data.get("sample_size", 0)

        # Collect normality test results for cross-referencing
        normality_results: Dict[str, bool] = {}
        for tr in test_results:
            inner = self._get_inner(tr)
            if inner.get("test_type") == "shapiro_wilk":
                var_name = tr.get("variable_name", "") or inner.get("variable_name", "")
                p_val = inner.get("p_value")
                if var_name and p_val is not None and isinstance(p_val, (int, float)):
                    normality_results[var_name] = p_val > 0.05  # True = normal

        # Collect unique categories per variable for group-count validation
        var_categories: Dict[str, int] = {}
        if df is not None:
            for var_name, var_info in variables.items():
                if var_info.get("dtype") == "categorical" and var_name in df.columns:
                    var_categories[var_name] = df[var_name].dropna().nunique()

        for tr in test_results:
            inner = self._get_inner(tr)
            test_type = inner.get("test_type", "") or tr.get("test_type", "")
            var_name = tr.get("variable_name", "") or inner.get("variable_name", "")
            if not test_type or not var_name:
                continue

            var_info = variables.get(var_name)
            if not var_info:
                continue

            dtype = var_info.get("dtype", "")

            # ── Check 1: Parametric test on non-normal data ──
            if test_type in ("unpaired_t_test", "paired_t_test", "one_way_anova"):
                if var_name in normality_results and not normality_results[var_name]:
                    non_param_alt = {
                        "unpaired_t_test": "Mann-Whitney U test",
                        "paired_t_test": "Wilcoxon signed-rank test",
                        "one_way_anova": "Kruskal-Wallis test",
                    }
                    items.append(ReviewItem(
                        check_id="parametric_on_nonnormal",
                        tier=2,
                        severity=Severity.ERROR,
                        category="Test Selection",
                        message=(
                            f"{var_name}: parametric test '{test_type}' used but "
                            f"Shapiro-Wilk indicates non-normal distribution"
                        ),
                        variable=var_name,
                        suggestion=f"Use {non_param_alt.get(test_type, 'non-parametric alternative')} instead",
                        auto_fixable=True,
                    ))
                elif var_name in normality_results and normality_results[var_name]:
                    items.append(ReviewItem(
                        check_id="parametric_normal_ok",
                        tier=2,
                        severity=Severity.OK,
                        category="Test Selection",
                        message=f"{var_name}: parametric '{test_type}' justified by normal distribution",
                        variable=var_name,
                    ))

            # ── Check 2: Non-parametric test on normal data (missed power) ──
            if test_type in ("mann_whitney_u", "kruskal_wallis"):
                if var_name in normality_results and normality_results[var_name] and sample_size >= 30:
                    param_alt = {
                        "mann_whitney_u": "unpaired t-test",
                        "kruskal_wallis": "one-way ANOVA",
                    }
                    items.append(ReviewItem(
                        check_id="nonparam_on_normal",
                        tier=2,
                        severity=Severity.INFO,
                        category="Test Selection",
                        message=(
                            f"{var_name}: non-parametric '{test_type}' used but data appears normal "
                            f"(N={sample_size}). Parametric test would have more statistical power."
                        ),
                        variable=var_name,
                        suggestion=f"Consider using {param_alt.get(test_type)} for higher power",
                    ))

            # ── Check 3: t-test used when >2 groups exist ──
            if test_type in ("unpaired_t_test", "paired_t_test") and df is not None:
                # Find the grouping variable (any categorical var with ≥3 categories)
                cat_vars = [
                    v for v in variables.values()
                    if v.get("dtype") == "categorical" and v["name"] != var_name
                ]
                for cv in cat_vars:
                    n_groups = var_categories.get(cv["name"], 0)
                    if n_groups > 2:
                        items.append(ReviewItem(
                            check_id="ttest_multigroup",
                            tier=2,
                            severity=Severity.WARNING,
                            category="Test Selection",
                            message=(
                                f"{var_name}: t-test used but grouping variable "
                                f"'{cv['name']}' has {n_groups} categories (>2). "
                                f"T-test only valid for exactly 2-group comparison."
                            ),
                            variable=var_name,
                            suggestion="Use one-way ANOVA (parametric) or Kruskal-Wallis (non-parametric) for ≥3 groups",
                            auto_fixable=True,
                        ))
                        break  # Only flag once per variable

            # ── Check 4: ANOVA used with only 2 groups ──
            if test_type == "one_way_anova" and df is not None:
                cat_vars = [
                    v for v in variables.values()
                    if v.get("dtype") == "categorical" and v["name"] != var_name
                ]
                for cv in cat_vars:
                    n_groups = var_categories.get(cv["name"], 0)
                    if n_groups == 2:
                        items.append(ReviewItem(
                            check_id="anova_two_groups",
                            tier=2,
                            severity=Severity.INFO,
                            category="Test Selection",
                            message=(
                                f"{var_name}: ANOVA used but '{cv['name']}' has only 2 groups. "
                                f"While ANOVA works, t-test is more standard for 2-group comparison."
                            ),
                            variable=var_name,
                            suggestion="Consider using unpaired t-test for 2-group comparison (equivalent result, conventional choice)",
                        ))
                        break

            # ── Check 5: Chi-square on very small expected counts → Fisher's exact ──
            if test_type == "chi_square" and df is not None and dtype == "categorical":
                if var_name in df.columns:
                    counts = df[var_name].dropna().value_counts()
                    n_categories = len(counts)
                    min_expected = sample_size / n_categories if n_categories > 0 else 0
                    if min_expected < 5 or (counts < 5).any():
                        items.append(ReviewItem(
                            check_id="chi_sq_should_fisher",
                            tier=2,
                            severity=Severity.WARNING,
                            category="Test Selection",
                            message=(
                                f"{var_name}: chi-square test used but expected cell frequencies <5. "
                                f"Chi-square approximation unreliable with sparse cells."
                            ),
                            variable=var_name,
                            suggestion="Use Fisher's exact test when any expected cell count < 5",
                            auto_fixable=True,
                        ))

            # ── Check 6: Z-test proportion appropriateness ──
            if test_type == "z_test_proportion":
                if dtype != "categorical":
                    items.append(ReviewItem(
                        check_id="ztest_not_categorical",
                        tier=2,
                        severity=Severity.ERROR,
                        category="Test Selection",
                        message=f"{var_name}: Z-test for proportion used on non-categorical data",
                        variable=var_name,
                        suggestion="Z-test for proportions only applies to categorical/binary variables",
                    ))
                elif dtype == "categorical" and var_name in var_categories:
                    n_cats = var_categories[var_name]
                    if n_cats > 5:
                        items.append(ReviewItem(
                            check_id="ztest_too_many_cats",
                            tier=2,
                            severity=Severity.INFO,
                            category="Test Selection",
                            message=(
                                f"{var_name}: Z-test proportion applied to variable with {n_cats} "
                                f"categories. Z-test is most meaningful for binary/few-category variables."
                            ),
                            variable=var_name,
                            suggestion="Consider chi-square goodness-of-fit test for multi-category variables",
                        ))

            # ── Check 7: Study design consistency ──
            if study_design == "single_cohort":
                two_group_tests = {"unpaired_t_test", "mann_whitney_u", "chi_square"}
                if test_type in two_group_tests:
                    items.append(ReviewItem(
                        check_id="study_design_mismatch",
                        tier=2,
                        severity=Severity.WARNING,
                        category="Test Selection",
                        message=(
                            f"{var_name}: comparison test '{test_type}' used in single-cohort study design. "
                            f"This requires defining comparison groups."
                        ),
                        variable=var_name,
                        suggestion="Verify that the study design should be 'two-group' or 'multi-group' instead of single cohort, or clarify which groups are being compared",
                    ))

            # ── Check 8: Small sample (N<30) with parametric test and no normality check ──
            if test_type in ("unpaired_t_test", "paired_t_test") and sample_size < 30:
                if var_name not in normality_results:
                    items.append(ReviewItem(
                        check_id="small_n_no_normality",
                        tier=2,
                        severity=Severity.WARNING,
                        category="Test Selection",
                        message=(
                            f"{var_name}: parametric '{test_type}' with small sample (N={sample_size}) "
                            f"but no normality test performed. Central Limit Theorem unreliable for N<30."
                        ),
                        variable=var_name,
                        suggestion="Run Shapiro-Wilk test first; if non-normal, switch to Mann-Whitney U / Wilcoxon",
                        auto_fixable=True,
                    ))

        return items

    def _check_sample_adequacy(self, data: dict, df: Optional[pd.DataFrame]) -> List[ReviewItem]:
        """Check if sample size is adequate for chosen tests."""
        items = []
        sample_size = data.get("sample_size", 0)
        test_results = data.get("test_results", [])

        min_sizes = {
            "unpaired_t_test": 30,
            "paired_t_test": 15,
            "one_way_anova": 20,
            "chi_square": 20,
        }

        for tr in test_results:
            inner = self._get_inner(tr)
            test_type = inner.get("test_type", "")
            min_n = min_sizes.get(test_type)

            if min_n and sample_size < min_n:
                items.append(ReviewItem(
                    check_id="sample_too_small",
                    tier=2,
                    severity=Severity.WARNING,
                    category="Sample Adequacy",
                    message=f"N={sample_size} may be insufficient for {test_type} (recommended ≥{min_n})",
                    suggestion="Consider non-parametric alternative or note as limitation",
                ))

        if not any(i.check_id == "sample_too_small" for i in items):
            items.append(ReviewItem(
                check_id="sample_adequate",
                tier=2,
                severity=Severity.OK,
                category="Sample Adequacy",
                message=f"Sample size N={sample_size} adequate for all tests used",
            ))
        return items

    def _check_normality_assumptions(self, data: dict, df: Optional[pd.DataFrame]) -> List[ReviewItem]:
        """Verify normality was tested before parametric tests."""
        items = []
        test_results = data.get("test_results", [])

        parametric_tests = {"unpaired_t_test", "paired_t_test", "one_way_anova"}
        parametric_vars = set()
        normality_tested = set()

        for tr in test_results:
            var_name = tr.get("variable_name", "")
            inner = self._get_inner(tr)
            test_type = inner.get("test_type", "")

            if test_type in parametric_tests:
                parametric_vars.add(var_name)
            if test_type == "shapiro_wilk":
                normality_tested.add(var_name)

        untested = parametric_vars - normality_tested
        if untested:
            for var in untested:
                items.append(ReviewItem(
                    check_id="normality_not_tested",
                    tier=2,
                    severity=Severity.WARNING,
                    category="Assumptions",
                    message=f"Parametric test used for '{var}' without documented normality check",
                    variable=var,
                    suggestion="Add Shapiro-Wilk test or state assumption justification",
                ))
        else:
            items.append(ReviewItem(
                check_id="normality_ok",
                tier=2,
                severity=Severity.OK,
                category="Assumptions",
                message="Normality assumptions verified for all parametric tests",
            ))
        return items

    def _check_multiple_testing(self, data: dict) -> List[ReviewItem]:
        """Flag if many comparisons without correction."""
        items = []
        test_results = data.get("test_results", [])

        significance_tests = [
            tr for tr in test_results
            if self._get_inner(tr).get("p_value") is not None
        ]

        if len(significance_tests) > 5:
            significant_count = sum(
                1 for tr in significance_tests
                if self._get_inner(tr).get("is_significant_05")
            )
            items.append(ReviewItem(
                check_id="multiple_testing",
                tier=2,
                severity=Severity.INFO,
                category="Multiple Testing",
                message=f"{len(significance_tests)} tests performed, {significant_count} significant. Consider Bonferroni correction.",
                suggestion=f"Adjusted α = {0.05 / len(significance_tests):.4f} (Bonferroni)",
            ))
        return items

    def _check_chi_square_assumptions(self, data: dict, df: Optional[pd.DataFrame]) -> List[ReviewItem]:
        """Check expected cell counts for chi-square tests."""
        items = []
        if df is None:
            return items

        variables = {v["name"]: v for v in data.get("variables", [])}
        test_results = data.get("test_results", [])

        for tr in test_results:
            inner = self._get_inner(tr)
            test_type = inner.get("test_type", "")
            var_name = tr.get("variable_name", "")

            if test_type != "chi_square" or not var_name:
                continue

            var_info = variables.get(var_name)
            if not var_info or var_info.get("dtype") != "categorical":
                continue

            if var_name in df.columns:
                value_counts = df[var_name].dropna().value_counts()
                min_count = value_counts.min() if len(value_counts) > 0 else 0

                # Expected count approximation: if any category has <5 observations
                if min_count < 5:
                    items.append(ReviewItem(
                        check_id="chi_sq_small_cell",
                        tier=2,
                        severity=Severity.WARNING,
                        category="Chi-square Assumptions",
                        message=f"{var_name}: smallest category has only {min_count} cases (expected ≥5 for chi-square)",
                        variable=var_name,
                        suggestion="Consider Fisher's exact test or merge sparse categories",
                        auto_fixable=True,
                    ))
        return items

    # ─── TIER 3: CLINICAL INTERPRETATION ──────────────────────────────────────

    async def _tier3_interpretation(self, data: dict, df: Optional[pd.DataFrame]) -> List[ReviewItem]:
        """LLM-powered interpretation checks (or heuristic fallback)."""
        items = []

        # 3.1 Statistical vs clinical significance
        items.extend(self._check_clinical_significance(data))

        # 3.2 Negative result interpretation
        items.extend(self._check_negative_results(data))

        # 3.3 Conclusion support (LLM if available)
        if self.services:
            tier3_llm = await self._llm_interpretation_check(data)
            items.extend(tier3_llm)

        return items

    def _check_clinical_significance(self, data: dict) -> List[ReviewItem]:
        """Flag cases where statistical significance may not equal clinical significance."""
        items = []
        test_results = data.get("test_results", [])
        sample_size = data.get("sample_size", 0)

        for tr in test_results:
            inner = self._get_inner(tr)
            p_value = inner.get("p_value")
            is_sig = inner.get("is_significant_05", False)
            var_name = tr.get("variable_name", "")

            if not p_value or not is_sig:
                continue
            if not isinstance(p_value, (int, float)):
                continue

            # Large samples can make trivial differences significant
            if sample_size > 200 and p_value > 0.01:
                items.append(ReviewItem(
                    check_id="clinical_vs_statistical",
                    tier=3,
                    severity=Severity.INFO,
                    category="Clinical Significance",
                    message=f"{var_name}: statistically significant (p={p_value:.4f}) with large N={sample_size}. Verify clinical relevance.",
                    variable=var_name,
                    suggestion="Report effect size and comment on clinical meaningfulness",
                ))
        return items

    def _check_negative_results(self, data: dict) -> List[ReviewItem]:
        """Check proper interpretation of non-significant results."""
        items = []
        sections = data.get("report_sections", [])

        problematic_phrases = [
            "no difference exists",
            "proved that there is no",
            "confirmed no association",
            "ruled out",
        ]

        for s in sections:
            content = s.get("content", "").lower()
            for phrase in problematic_phrases:
                if phrase in content:
                    items.append(ReviewItem(
                        check_id="negative_overstatement",
                        tier=3,
                        severity=Severity.WARNING,
                        category="Interpretation",
                        message=f"Potential overstatement of negative result: '{phrase}' found",
                        suggestion="Non-significant p-value means 'insufficient evidence to reject H0', not 'no difference'",
                    ))
                    break

        # Check borderline p-values
        test_results = data.get("test_results", [])
        for tr in test_results:
            inner = self._get_inner(tr)
            p_value = inner.get("p_value")
            var_name = tr.get("variable_name", "")
            if p_value and isinstance(p_value, (int, float)) and 0.05 < p_value < 0.10:
                items.append(ReviewItem(
                    check_id="borderline_p",
                    tier=3,
                    severity=Severity.INFO,
                    category="Interpretation",
                    message=f"{var_name}: borderline p-value ({p_value:.4f}). Consider noting as 'trend'.",
                    variable=var_name,
                    suggestion="Mention as 'approaching significance' or 'trend towards significance (p=X)'",
                ))
        return items

    async def _llm_interpretation_check(self, data: dict) -> List[ReviewItem]:
        """Use LLM to check if conclusions are supported by evidence."""
        items = []
        sections = data.get("report_sections", [])
        title = data.get("title", "")

        # Collect all test results and narrative for LLM review
        narratives = [s["content"] for s in sections if s.get("type") == "paragraph"]
        test_summaries = []
        for tr in data.get("test_results", []):
            inner = self._get_inner(tr)
            if inner.get("significance_statement"):
                test_summaries.append(f"- {tr.get('variable_name')}: {inner['significance_statement']}")

        if not narratives or not test_summaries:
            return items

        prompt = f"""Review this medical thesis results section for logical consistency.

Title: {title}
Test Results:
{chr(10).join(test_summaries[:10])}

Narrative excerpts:
{chr(10).join(narratives[:5])}

Check for:
1. Any conclusion not supported by the statistical results
2. Any claim that overstates the evidence
3. Any important finding that is not mentioned in the narrative

Respond in JSON format:
{{"issues": [{{"variable": "...", "issue": "...", "suggestion": "..."}}], "overall": "ok|minor_issues|significant_issues"}}"""

        try:
            response = await self.services.generate_json(
                prompt=prompt,
                system_prompt="You are a biostatistics reviewer. Be concise. Only flag genuine issues.",
                temperature=0.2,
            )
            if response and response.get("issues"):
                for issue in response["issues"][:5]:
                    items.append(ReviewItem(
                        check_id="llm_interpretation",
                        tier=3,
                        severity=Severity.WARNING,
                        category="Interpretation (AI)",
                        message=issue.get("issue", ""),
                        variable=issue.get("variable", ""),
                        suggestion=issue.get("suggestion", ""),
                    ))
        except Exception as e:
            logger.warning("LLM interpretation check failed", error=str(e))

        return items
