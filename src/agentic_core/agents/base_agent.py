"""
Backward-compatible re-export module.

Canonical locations:
  - BaseAgent: src.agentic_core.agents.base
  - MissionState: src.agentic_core.agents.state
  - AgentContext, AgentPlan, AgentAction, AgentPhase: src.agentic_core.agents.context
"""
from src.agentic_core.agents.base import BaseAgent  # noqa: F401
from src.agentic_core.agents.state import MissionState  # noqa: F401
from src.agentic_core.agents.context import (  # noqa: F401
    AgentPhase,
    AgentContext,
    AgentPlan,
    AgentAction,
)
