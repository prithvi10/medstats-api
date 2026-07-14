"""
Test Selector Agent — Schemas and data models.
"""
from dataclasses import dataclass, field
from typing import Any, Dict

from src.medstats.core.stats_engine import TestType


@dataclass
class TestSelection:
    """Represents a selected statistical test and its justification."""

    variable_name: str
    test_type: TestType
    justification: str
    params: Dict[str, Any] = field(default_factory=dict)
