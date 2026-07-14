"""
Document Parser Agent — Schemas and data models.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class VariableInfo:
    """Metadata about a single variable in the dataset."""

    name: str
    dtype: str  # "numerical" | "categorical"
    unique_values: int = 0
    missing_count: int = 0
    categories: List[str] = field(default_factory=list)


@dataclass
class ParsedDataset:
    """Output of the document parsing stage."""

    dataframe: Any = None  # pd.DataFrame
    dataframe_json: str = ""
    variables: List[VariableInfo] = field(default_factory=list)
    study_design: str = "single_cohort"
    groups: Optional[Dict[str, Any]] = None
    sample_size: int = 0
    missing_data_report: Dict[str, int] = field(default_factory=dict)
    intent: str = ""
    title: str = ""
