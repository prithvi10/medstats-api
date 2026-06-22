from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum
from src.shared.logging.logger import get_logger


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


class BaseAgent(ABC):
    """
    Abstract base class implementing the 4-phase agentic loop:
    PERCEPTION -> REASONING -> ACTION -> FEEDBACK

    Cost optimization: reasoning is deterministic by default.
    Only override _reason_with_llm() if complex multi-step planning is needed.
    """

    role_name: str = "BaseAgent"
    requires_llm_reasoning: bool = False

    def __init__(self, services=None):
        self.services = services
        self.logger = get_logger(f"agent.{self.role_name}")

    async def run(self, state: MissionState) -> MissionState:
        """Execute the full agent pipeline."""
        try:
            state.add_log(f"{self.role_name}: Perceiving...")
            context = await self.perceive(state)

            state.add_log(f"{self.role_name}: Reasoning...")
            plan = await self.reason(state, context)

            state.add_log(f"{self.role_name}: Acting...")
            actions, state = await self.act(state, context, plan)

            state.add_log(f"{self.role_name}: Feedback...")
            await self.feedback(state, actions)

            state.add_log(f"{self.role_name}: Complete.")
            return state

        except Exception as e:
            self.logger.error("Agent failed", agent=self.role_name, error=str(e))
            state.set_error(f"{self.role_name} failed: {str(e)}")
            return state

    async def perceive(self, state: MissionState) -> AgentContext:
        """Phase 1: Gather context from environment and state."""
        context = AgentContext(raw_input=state.agent_outputs)
        domain_context = await self._perceive_domain(state, context)
        return domain_context or context

    async def reason(self, state: MissionState, context: AgentContext) -> AgentPlan:
        """Phase 2: Deterministic planning by default."""
        if self.requires_llm_reasoning:
            return await self._reason_with_llm(state, context)
        return self._create_default_plan(state, context)

    async def act(self, state: MissionState, context: AgentContext, plan: AgentPlan):
        """Phase 3: Execute the plan."""
        actions, new_state = await self._act_domain(state, context, plan)
        return actions, new_state

    async def feedback(self, state: MissionState, actions: List[AgentAction]):
        """Phase 4: Learn from outcomes (optional override)."""
        pass

    @abstractmethod
    async def _perceive_domain(self, state: MissionState, context: AgentContext) -> AgentContext:
        """Domain-specific context gathering. Must be implemented."""
        pass

    def _create_default_plan(self, state: MissionState, context: AgentContext) -> AgentPlan:
        """Default deterministic plan. Override for custom logic."""
        return AgentPlan(steps=["execute"], selected_tools=["default"], confidence=1.0)

    async def _reason_with_llm(self, state: MissionState, context: AgentContext) -> AgentPlan:
        """LLM-based reasoning. Override if requires_llm_reasoning=True."""
        return self._create_default_plan(state, context)

    @abstractmethod
    async def _act_domain(self, state: MissionState, context: AgentContext, plan: AgentPlan):
        """Domain-specific action execution. Must be implemented."""
        pass
