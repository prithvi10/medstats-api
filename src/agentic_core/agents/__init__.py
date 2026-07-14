"""
Agentic Core — Domain-agnostic agent framework.

Public API:
    BaseAgent       — Abstract base for all agents
    MissionState    — Shared pipeline state
    AgentContext    — Perception output
    AgentPlan       — Reasoning output
    AgentAction     — Action result
    MissionControl  — Pipeline orchestrator
"""
from src.agentic_core.agents.base import BaseAgent  # noqa: F401
from src.agentic_core.agents.state import MissionState  # noqa: F401
from src.agentic_core.agents.context import (  # noqa: F401
    AgentPhase,
    AgentContext,
    AgentPlan,
    AgentAction,
)
from src.agentic_core.agents.orchestrator import MissionControl  # noqa: F401
