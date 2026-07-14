"""
Agent context, plan, and action dataclasses.

These represent the data structures flowing through the agentic loop:
  AgentContext — gathered during PERCEPTION
  AgentPlan    — produced during REASONING
  AgentAction  — result of ACTION execution
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum


class AgentPhase(str, Enum):
    PERCEPTION = "perception"
    REASONING = "reasoning"
    ACTION = "action"
    FEEDBACK = "feedback"


@dataclass
class AgentContext:
    """Context gathered during perception phase."""

    raw_input: Dict[str, Any] = field(default_factory=dict)
    parsed_data: Any = None
    external_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentPlan:
    """Plan produced during reasoning phase."""

    steps: List[str] = field(default_factory=list)
    selected_tools: List[str] = field(default_factory=list)
    confidence: float = 1.0
    reasoning: str = ""


@dataclass
class AgentAction:
    """Result of an action execution."""

    tool_name: str = ""
    input_params: Dict[str, Any] = field(default_factory=dict)
    output: Any = None
    success: bool = True
    error: Optional[str] = None

    @classmethod
    def success_action(cls, tool_name: str, output: Any, **params):
        return cls(tool_name=tool_name, input_params=params, output=output, success=True)

    @classmethod
    def failure_action(cls, tool_name: str, error: str, **params):
        return cls(tool_name=tool_name, input_params=params, success=False, error=error)
