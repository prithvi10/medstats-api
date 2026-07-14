"""
MissionState — Shared state clipboard passed between agents in a pipeline.

This is the single source of truth for pipeline execution state.
All agents read/write to `agent_outputs` dict as a shared clipboard.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class MissionState:
    """Shared state clipboard passed between agents in a pipeline."""

    submission_id: str = ""
    job_id: str = ""
    status: str = "PENDING"
    current_agent_index: int = 0
    logs: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    agent_outputs: Dict[str, Any] = field(default_factory=dict)
    progress: int = 0

    def add_log(self, message: str):
        self.logs.append(message)

    def set_error(self, message: str):
        self.error_message = message
        self.status = "FAILED"

    def to_dict(self) -> dict:
        return {
            "submission_id": self.submission_id,
            "job_id": self.job_id,
            "status": self.status,
            "current_agent_index": self.current_agent_index,
            "logs": self.logs,
            "error_message": self.error_message,
            "progress": self.progress,
        }
