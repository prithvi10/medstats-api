"""
Review Agent — Schemas and data models.
"""
from dataclasses import dataclass, field, asdict
from typing import List
from enum import Enum


class Severity(str, Enum):
    OK = "ok"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class ReviewItem:
    """A single review finding."""

    check_id: str
    tier: int
    severity: Severity
    category: str
    message: str
    variable: str = ""
    suggestion: str = ""
    citation: str = ""
    auto_fixable: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ReviewSummary:
    """Aggregated review output."""

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
