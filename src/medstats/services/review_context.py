"""Build professor review context and data profiles from pipeline state."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.medstats.db.models import (
    AnalysisJob,
    JobStatus,
    ProfessorReview,
    ProfessorStudent,
    Report,
    Submission,
    User,
)
from src.medstats.services.review_presets import REVIEW_PRESETS, VALID_TESTS_BY_DESIGN, ALL_TEST_OPTIONS


STUDY_DESIGN_LABELS = {
    "single_cohort": "Single cohort / descriptive",
    "two_group": "Two-group comparative",
    "multi_group": "Multi-group comparative",
}

# Descriptive proportion test — deprioritize when a variable also has an inferential row.
DESCRIPTIVE_PROPORTION_TEST = "z_test_proportion"

CATEGORICAL_TEST_VALUES = {"z_test_proportion", "chi_square", "fisher_exact"}
NUMERICAL_TESTS_BY_DESIGN = {
    "single_cohort": {"shapiro_wilk", "z_test_proportion"},
    "two_group": {"unpaired_t_test", "paired_t_test", "mann_whitney_u", "shapiro_wilk"},
    "multi_group": {"one_way_anova", "kruskal_wallis", "shapiro_wilk"},
}


async def get_linked_student_ids(db: AsyncSession, professor_id: str) -> List[str]:
    result = await db.execute(
        select(ProfessorStudent.student_id).where(ProfessorStudent.professor_id == professor_id)
    )
    return [str(row[0]) for row in result.all()]


async def professor_can_access_submission(
    db: AsyncSession, professor_id: str, submission: Submission
) -> bool:
    if submission.professor_id and str(submission.professor_id) == professor_id:
        return True
    linked = await get_linked_student_ids(db, professor_id)
    return str(submission.student_id) in linked


async def get_submission_for_professor(
    db: AsyncSession, professor_id: str, submission_id
) -> Optional[Submission]:
    result = await db.execute(select(Submission).where(Submission.id == submission_id))
    submission = result.scalar_one_or_none()
    if not submission:
        return None
    if not await professor_can_access_submission(db, professor_id, submission):
        return None
    return submission


async def list_professor_submissions(db: AsyncSession, professor_id: str) -> List[Submission]:
    linked_ids = await get_linked_student_ids(db, professor_id)
    conditions = [Submission.professor_id == professor_id]
    if linked_ids:
        conditions.append(Submission.student_id.in_(linked_ids))
    result = await db.execute(
        select(Submission)
        .where(or_(*conditions))
        .order_by(Submission.updated_at.desc())
    )
    return list(result.scalars().all())


async def get_latest_analysis_state(db: AsyncSession, submission_id) -> Dict[str, Any]:
    """Load agent_state aligned with the latest report when possible."""
    report = await get_latest_report(db, submission_id)
    if report and report.analysis_job_id:
        job_result = await db.execute(
            select(AnalysisJob).where(AnalysisJob.id == report.analysis_job_id)
        )
        job = job_result.scalar_one_or_none()
        if job and job.agent_state:
            state = json.loads(job.agent_state) if isinstance(job.agent_state, str) else job.agent_state
            if state.get("variables") or state.get("test_results"):
                return state

    result = await db.execute(
        select(AnalysisJob)
        .where(AnalysisJob.submission_id == submission_id)
        .where(AnalysisJob.status == JobStatus.COMPLETED)
        .order_by(AnalysisJob.completed_at.desc())
    )
    jobs = result.scalars().all()
    for job in jobs:
        if not job.agent_state:
            continue
        state = json.loads(job.agent_state) if isinstance(job.agent_state, str) else job.agent_state
        if state.get("variables") or state.get("test_results"):
            return state
    return {}


async def count_review_rounds(db: AsyncSession, submission_id) -> int:
    report_ids = await db.execute(
        select(Report.id).where(Report.submission_id == submission_id)
    )
    ids = [row[0] for row in report_ids.all()]
    if not ids:
        return 0
    result = await db.execute(
        select(ProfessorReview).where(ProfessorReview.report_id.in_(ids))
    )
    return len(result.scalars().all())


def _distribution_note(var_info: Dict[str, Any]) -> str:
    dtype = var_info.get("dtype", "")
    if dtype == "categorical":
        n_cat = var_info.get("n_categories") or var_info.get("unique_count")
        if n_cat == 2:
            return "binary"
        return f"categorical ({n_cat or '?'} levels)" if n_cat else "categorical"
    if dtype == "numerical":
        return "continuous numerical"
    return dtype or "unknown"


def _group_test_results_by_variable(test_results: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for tr in test_results:
        var_name = tr.get("variable_name", "")
        if not var_name:
            continue
        grouped.setdefault(var_name, []).append(tr)
    return grouped


def _resolve_current_test_for_variable(rows: List[Dict[str, Any]]) -> Optional[str]:
    """
    Pick the inferential test for a variable when pipeline state has duplicate rows
    (descriptive row with z_test_proportion + separate inferential row).
    """
    candidates: List[str] = []
    for tr in rows:
        inner = tr.get("test_result") or {}
        if not isinstance(inner, dict):
            continue
        test_type = inner.get("test_type")
        if test_type and test_type != "shapiro_wilk":
            candidates.append(test_type)

    if not candidates:
        return None

    non_descriptive = [t for t in candidates if t != DESCRIPTIVE_PROPORTION_TEST]
    if non_descriptive:
        return non_descriptive[-1]
    return candidates[-1]


def _extract_normality_by_variable(test_results: List[Dict[str, Any]]) -> Dict[str, bool]:
    normality_by_var: Dict[str, bool] = {}
    for tr in test_results:
        var_name = tr.get("variable_name", "")
        inner = tr.get("test_result") or {}
        if not isinstance(inner, dict):
            continue
        if inner.get("test_type") != "shapiro_wilk":
            continue
        p_val = inner.get("p_value")
        if p_val is not None:
            normality_by_var[var_name] = p_val > 0.05
    return normality_by_var


def build_data_profile(agent_state: Dict[str, Any]) -> Dict[str, Any]:
    variables = agent_state.get("variables", [])
    test_results = agent_state.get("test_results", [])
    study_design = agent_state.get("study_design", "unknown")
    sample_size = agent_state.get("sample_size", 0)

    grouped_results = _group_test_results_by_variable(test_results)
    normality_by_var = _extract_normality_by_variable(test_results)
    test_by_var: Dict[str, str] = {}
    for var_name, rows in grouped_results.items():
        current = _resolve_current_test_for_variable(rows)
        if current:
            test_by_var[var_name] = current

    var_profiles = []
    parametric_count = 0
    for v in variables:
        name = v.get("name", "")
        current_test = test_by_var.get(name)
        if current_test in ("unpaired_t_test", "paired_t_test", "one_way_anova"):
            parametric_count += 1
        var_profiles.append({
            "name": name,
            "dtype": v.get("dtype"),
            "groups": v.get("groups"),
            "current_test": current_test,
            "normality_ok": normality_by_var.get(name),
            "missing_pct": v.get("missing_pct"),
            "distribution_note": _distribution_note(v),
            "valid_test_alternatives": _valid_tests_for_variable(v, study_design, current_test),
        })

    design_label = STUDY_DESIGN_LABELS.get(study_design, study_design.replace("_", " ").title())
    normality_tested = len(normality_by_var)
    summary_parts = [
        f"{design_label} study with N={sample_size} and {len(variables)} variable(s).",
    ]
    if parametric_count:
        summary_parts.append(
            f"{parametric_count} parametric test(s) applied"
            + (f"; normality assessed for {normality_tested} variable(s)." if normality_tested else ".")
        )
    else:
        summary_parts.append("Primarily non-parametric or descriptive analyses.")

    return {
        "study_design": study_design,
        "study_design_label": design_label,
        "sample_size": sample_size,
        "variable_count": len(variables),
        "variables": var_profiles,
        "summary": " ".join(summary_parts),
        "test_options": ALL_TEST_OPTIONS,
        "presets": REVIEW_PRESETS,
    }


def _valid_tests_for_variable(
    var_info: Dict[str, Any], study_design: str, current_test: Optional[str]
) -> List[Dict[str, str]]:
    dtype = var_info.get("dtype", "")
    if dtype == "categorical":
        allowed = set(CATEGORICAL_TEST_VALUES)
    elif dtype == "numerical":
        allowed = set(NUMERICAL_TESTS_BY_DESIGN.get(study_design, set()))
        if not allowed:
            allowed = set().union(*NUMERICAL_TESTS_BY_DESIGN.values())
    else:
        allowed = {t["value"] for t in ALL_TEST_OPTIONS}

    options = [t for t in ALL_TEST_OPTIONS if t["value"] in allowed]
    if current_test and not any(o["value"] == current_test for o in options):
        match = next((t for t in ALL_TEST_OPTIONS if t["value"] == current_test), None)
        if match:
            options.insert(0, match)
    return options


def build_master_chart_preview(
    agent_state: Dict[str, Any],
    submission_files: list,
    *,
    max_rows: int = 50,
) -> Optional[Dict[str, Any]]:
    """Build scrollable preview table for the student's master chart / data file."""
    import os
    from io import StringIO
    import pandas as pd

    file_record = _pick_data_file(submission_files)
    file_meta = {
        "file_id": str(file_record.id) if file_record else None,
        "file_name": file_record.original_name if file_record else "Master chart",
        "file_type": file_record.file_type if file_record else "data_excel",
    }

    df = None
    if agent_state.get("dataframe_json"):
        try:
            df = pd.read_json(StringIO(agent_state["dataframe_json"]))
        except Exception:
            df = None

    if df is None and file_record and os.path.isfile(file_record.file_url):
        df = _read_dataframe_from_path(file_record.file_url, file_record.original_name)

    if df is None or df.empty:
        return None

    if len(df) > 0 and df.iloc[0].isna().sum() > len(df.columns) * 0.3:
        df = df.iloc[1:].reset_index(drop=True)

    total_rows = len(df)
    preview_df = df.head(max_rows)
    headers = [str(c) for c in preview_df.columns.tolist()]
    rows = [
        [_cell_str(v) for v in row]
        for row in preview_df.values.tolist()
    ]

    return {
        **file_meta,
        "row_count": total_rows,
        "column_count": len(df.columns),
        "headers": headers,
        "rows": rows,
        "truncated": total_rows > max_rows,
    }


def _pick_data_file(files: list):
    for f in files:
        name = (getattr(f, "original_name", "") or "").lower()
        ftype = getattr(f, "file_type", "") or ""
        if ftype in ("data_excel", "data_csv"):
            return f
        if name.endswith((".xlsx", ".xls", ".csv")):
            return f
    for f in files:
        name = (getattr(f, "original_name", "") or "").lower()
        if any(k in name for k in ("master", "chart", "data")):
            return f
    return files[0] if files else None


def _read_dataframe_from_path(path: str, name: str):
    import pandas as pd
    lower = name.lower()
    try:
        if lower.endswith(".csv"):
            return pd.read_csv(path)
        if lower.endswith((".xlsx", ".xls")):
            df = pd.read_excel(path, sheet_name=0, header=0)
            if len(df) > 0 and df.iloc[0].isna().sum() > len(df.columns) * 0.3:
                df = df.iloc[1:].reset_index(drop=True)
            return df
        if lower.endswith(".docx"):
            from src.medstats.agents.document_parser.agent import DocumentParserAgent
            parser = DocumentParserAgent(services=None)
            return parser._parse_docx_tables(path)
    except Exception:
        return None
    return None


def _cell_str(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        import math
        if math.isnan(value):
            return ""
    text = str(value)
    return "" if text in ("nan", "None") else text


def build_tests_selected(data_profile: Optional[Dict[str, Any]]) -> List[Dict[str, str]]:
    if not data_profile:
        return []
    tests = []
    for v in data_profile.get("variables", []):
        if v.get("current_test"):
            label = next(
                (t["label"] for t in ALL_TEST_OPTIONS if t["value"] == v["current_test"]),
                v["current_test"].replace("_", " "),
            )
            tests.append({"variable": v["name"], "test": v["current_test"], "label": label})
    return tests


async def get_latest_report(db: AsyncSession, submission_id) -> Optional[Report]:
    """Return the highest-version report for a submission."""
    result = await db.execute(
        select(Report)
        .where(Report.submission_id == submission_id)
        .order_by(Report.version.desc(), Report.generated_at.desc())
        .limit(1)
    )
    return result.scalars().first()


async def build_review_history(db: AsyncSession, submission_id) -> List[Dict[str, Any]]:
    reports_result = await db.execute(
        select(Report)
        .where(Report.submission_id == submission_id)
        .order_by(Report.version.asc())
    )
    reports = reports_result.scalars().all()
    history = []
    for report in reports:
        reviews_result = await db.execute(
            select(ProfessorReview, User)
            .join(User, ProfessorReview.professor_id == User.id)
            .where(ProfessorReview.report_id == report.id)
            .order_by(ProfessorReview.reviewed_at.asc())
        )
        for review, professor in reviews_result.all():
            history.append({
                "id": str(review.id),
                "report_version": report.version,
                "decision": review.decision.value,
                "feedback": review.feedback,
                "section_comments": review.section_comments,
                "professor_name": professor.name,
                "reviewed_at": review.reviewed_at.isoformat(),
            })
    return history
