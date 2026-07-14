"""
Document Parser Agent — Data normalizers and cleaning utilities.

Handles case-insensitive deduplication, whitespace collapse,
and type coercion for messy real-world medical data.
"""
import re
from typing import Optional

import pandas as pd


def normalize_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize categorical columns: case-insensitive dedup, strip whitespace.
    'male', 'Male', 'MALE', ' male ' all become 'Male'.
    """
    for col in df.columns:
        series = df[col]
        if not pd.api.types.is_string_dtype(series) and not pd.api.types.is_object_dtype(series):
            continue
        numeric_check = pd.to_numeric(series, errors="coerce")
        if numeric_check.notna().sum() > len(series) * 0.5:
            continue

        cleaned = series.dropna()
        if cleaned.empty:
            continue

        case_map = {}
        for val in cleaned:
            s = str(val).strip()
            s = re.sub(r'\s+', ' ', s)
            if not s:
                continue
            key = s.lower()
            if key not in case_map:
                case_map[key] = s.title()

        df[col] = series.apply(
            lambda x: case_map.get(
                re.sub(r'\s+', ' ', str(x).strip()).lower(), x
            ) if pd.notna(x) and str(x).strip() else x
        )

    return df


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attempt to convert string columns to numeric where possible.
    Used as fallback when openpyxl reads everything as strings.
    """
    for col in df.columns:
        if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().sum() > len(df[col]) * 0.7:
                df[col] = converted
    return df
