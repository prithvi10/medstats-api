"""
Document Parser Agent — Ingests uploaded files and extracts structured data.

Handles Excel, CSV, and DOCX formats with varying schemas.
Normalizes categories, classifies variable types, and detects study design.
"""
import re
from typing import Any, List, Dict, Optional

import pandas as pd

from src.agentic_core.agents.base import BaseAgent
from src.agentic_core.agents.state import MissionState
from src.agentic_core.agents.context import AgentContext, AgentPlan, AgentAction
from src.medstats.agents.document_parser.schemas import ParsedDataset, VariableInfo
from src.medstats.agents.document_parser.normalizers import normalize_categories, coerce_types
from src.shared.logging.logger import get_logger

logger = get_logger(__name__)

SKIP_COLUMNS = {"no", "s.no", "s.no.", "sr", "sr.", "name", "ipd", "patient id", "date"}


class DocumentParserAgent(BaseAgent):
    role_name = "DocumentParser"
    requires_llm_reasoning = False

    async def _perceive_domain(self, state: MissionState, context: AgentContext) -> AgentContext:
        """Load files from disk and extract raw data."""
        files = state.agent_outputs.get("files", [])
        context.raw_input = {"files": files}

        dataframes = []
        supporting_text = ""

        for file_info in files:
            path = file_info["path"]
            file_type = file_info["type"]

            if file_type in ("data_excel",) and path.endswith((".xlsx", ".xls")):
                df = self._parse_excel(path)
                if df is not None:
                    dataframes.append(df)
            elif file_type == "data_csv" and path.endswith(".csv"):
                df = pd.read_csv(path)
                dataframes.append(df)
            elif file_type == "supporting_doc" and path.endswith(".docx"):
                text = self._parse_docx(path)
                supporting_text += text + "\n"
            elif path.endswith((".xlsx", ".xls")):
                df = self._parse_excel(path)
                if df is not None:
                    dataframes.append(df)
            elif path.endswith(".docx"):
                text = self._parse_docx(path)
                if "master" in path.lower() or "chart" in path.lower() or "data" in path.lower():
                    df = self._parse_docx_tables(path)
                    if df is not None:
                        dataframes.append(df)
                else:
                    supporting_text += text + "\n"

        context.parsed_data = {
            "dataframes": dataframes,
            "supporting_text": supporting_text,
        }
        return context

    async def _act_domain(self, state: MissionState, context: AgentContext, plan: AgentPlan):
        """Process parsed data into structured dataset."""
        data = context.parsed_data
        dataframes = data.get("dataframes", [])
        supporting_text = data.get("supporting_text", "")
        intent = state.agent_outputs.get("intent", "")

        if not dataframes:
            return [AgentAction.failure_action("parse", "No data files found")], state

        if len(dataframes) > 1 and all(len(df) == len(dataframes[0]) for df in dataframes):
            merged = pd.concat(dataframes, axis=1)
        else:
            merged = dataframes[0]

        merged.columns = [str(c).strip() for c in merged.columns]
        merged = merged.dropna(how="all").dropna(axis=1, how="all")
        merged = normalize_categories(merged)

        variables = self._classify_variables(merged)
        study_design, groups = self._detect_study_design(merged, variables, supporting_text)

        parsed = ParsedDataset(
            dataframe=merged,
            dataframe_json=merged.to_json(),
            variables=variables,
            study_design=study_design,
            groups=groups,
            sample_size=len(merged),
            missing_data_report={v.name: v.missing_count for v in variables if v.missing_count > 0},
            intent=intent or supporting_text[:500],
            title=state.agent_outputs.get("title", ""),
        )

        state.agent_outputs["parsed_dataset"] = parsed
        state.agent_outputs["dataframe_json"] = merged.to_json()
        state.agent_outputs["variables"] = [vars(v) for v in variables]
        state.agent_outputs["study_design"] = study_design
        state.agent_outputs["sample_size"] = len(merged)
        state.agent_outputs["groups"] = groups

        action = AgentAction.success_action(
            "parse",
            output=f"Parsed {len(merged)} records, {len(variables)} variables, design: {study_design}",
        )
        return [action], state

    # ------------------------------------------------------------------
    # File parsers
    # ------------------------------------------------------------------

    def _parse_excel(self, path: str) -> Optional[pd.DataFrame]:
        try:
            df = pd.read_excel(path, sheet_name=0, header=0)
            if len(df) > 0 and df.iloc[0].isna().sum() > len(df.columns) * 0.5:
                df = df.iloc[1:].reset_index(drop=True)
            return df
        except (OverflowError, Exception) as e:
            if "Overflow" in str(e) or "datetime" in str(e).lower():
                try:
                    from openpyxl import load_workbook
                    wb = load_workbook(path, data_only=True)
                    ws = wb.active
                    data = list(ws.values)
                    if len(data) < 2:
                        return None
                    headers = [str(h) if h else f"Col_{i}" for i, h in enumerate(data[0])]
                    rows = [[str(cell) if cell is not None else None for cell in row] for row in data[1:]]
                    df = pd.DataFrame(rows, columns=headers)
                    df = coerce_types(df)
                    return df
                except Exception as e2:
                    logger.error("Excel parse failed (openpyxl fallback)", path=path, error=str(e2))
                    return None
            logger.error("Excel parse failed", path=path, error=str(e))
            return None

    def _parse_docx(self, path: str) -> str:
        try:
            from docx import Document
            doc = Document(path)
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        except Exception as e:
            logger.error("DOCX parse failed", path=path, error=str(e))
            return ""

    def _parse_docx_tables(self, path: str) -> Optional[pd.DataFrame]:
        """Extract tables from a DOCX file (handles MASTERCHART.docx pattern)."""
        try:
            from docx import Document
            doc = Document(path)
            all_dfs = []

            for table in doc.tables:
                rows = []
                for row in table.rows:
                    cells = [cell.text.strip() for cell in row.cells]
                    rows.append(cells)

                if len(rows) < 2:
                    continue

                df = pd.DataFrame(rows[1:], columns=rows[0])
                all_dfs.append(df)

            if not all_dfs:
                return None

            if len(all_dfs) > 1 and all(len(df) == len(all_dfs[0]) for df in all_dfs):
                return pd.concat(all_dfs, axis=1)
            return all_dfs[0]

        except Exception as e:
            logger.error("DOCX table parse failed", path=path, error=str(e))
            return None

    # ------------------------------------------------------------------
    # Variable classification
    # ------------------------------------------------------------------

    def _classify_variables(self, df: pd.DataFrame) -> List[VariableInfo]:
        """Classify each column by data type."""
        variables = []

        for col in df.columns:
            if col.lower().strip() in SKIP_COLUMNS:
                continue

            series = df[col]
            unique = series.nunique()
            missing = int(series.isna().sum())

            numeric_series = pd.to_numeric(series, errors="coerce")
            numeric_ratio = numeric_series.notna().sum() / max(len(series), 1)

            if numeric_ratio > 0.8 and unique > 10:
                dtype = "numerical"
                categories = []
            elif unique <= 20:
                dtype = "categorical"
                categories = [str(v) for v in series.dropna().unique()[:20]]
            else:
                dtype = "numerical" if numeric_ratio > 0.5 else "categorical"
                categories = []

            variables.append(VariableInfo(
                name=col,
                dtype=dtype,
                unique_values=unique,
                missing_count=missing,
                categories=categories,
            ))

        return variables

    # ------------------------------------------------------------------
    # Study design detection
    # ------------------------------------------------------------------

    def _detect_study_design(
        self, df: pd.DataFrame, variables: List[VariableInfo], context: str
    ) -> tuple:
        """Detect whether this is single-cohort, two-group, or multi-group study."""
        context_lower = context.lower()

        group_keywords = ["group", "arm", "intervention", "control", "treatment"]
        for var in variables:
            if any(kw in var.name.lower() for kw in group_keywords):
                if var.unique_values == 2:
                    return "two_group", {"column": var.name, "groups": var.categories}
                elif var.unique_values > 2:
                    return "multi_group", {"column": var.name, "groups": var.categories}

        if any(kw in context_lower for kw in ["randomized", "randomised", "rct", "group a", "group b", "group l"]):
            return "two_group", None
        if any(kw in context_lower for kw in ["cohort", "prospective", "retrospective", "observational"]):
            return "single_cohort", None

        return "single_cohort", None
