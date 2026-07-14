"""
BaseAgent — Abstract base class implementing the 4-phase agentic loop.

PERCEPTION -> REASONING -> ACTION -> FEEDBACK

Cost optimization: reasoning is deterministic by default.
Only override _reason_with_llm() if complex multi-step planning is needed.
"""
from abc import ABC, abstractmethod
from typing import List

from src.agentic_core.agents.state import MissionState
from src.agentic_core.agents.context import AgentContext, AgentPlan, AgentAction
from src.shared.logging.logger import get_logger


class BaseAgent(ABC):
    """
    Abstract base class implementing the 4-phase agentic loop.

    Subclasses MUST implement:
      - _perceive_domain(state, context) -> AgentContext
      - _act_domain(state, context, plan) -> (List[AgentAction], MissionState)

    Subclasses MAY override:
      - _create_default_plan(state, context) -> AgentPlan
      - _reason_with_llm(state, context) -> AgentPlan
      - feedback(state, actions) -> None
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
