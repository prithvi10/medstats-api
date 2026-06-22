"""
MissionControl - Pipeline orchestrator for multi-agent workflows.
Executes agents in sequence, tracks progress, and yields state updates.
"""
from typing import List, Type, AsyncGenerator, Optional, Callable
from src.agentic_core.agents.base_agent import BaseAgent, MissionState
from src.shared.logging.logger import get_logger

logger = get_logger(__name__)


class MissionControl:
    """
    Orchestrates a pipeline of agents for a single analysis job.
    Executes sequentially, updating state after each agent.
    """

    def __init__(
        self,
        agents: List[BaseAgent],
        progress_callback: Optional[Callable] = None,
    ):
        self.agents = agents
        self.progress_callback = progress_callback

    async def execute(self, state: MissionState) -> AsyncGenerator[MissionState, None]:
        """Execute full pipeline, yielding state after each agent completes."""
        total = len(self.agents)
        state.status = "RUNNING"

        for i, agent in enumerate(self.agents):
            state.current_agent_index = i
            state.progress = int((i / total) * 100)
            state.add_log(f"Pipeline step {i+1}/{total}: {agent.role_name}")

            if self.progress_callback:
                await self.progress_callback(state)

            yield state

            state = await agent.run(state)

            if state.status == "FAILED":
                logger.error("Pipeline failed", agent=agent.role_name, error=state.error_message)
                yield state
                return

            state.agent_outputs[agent.role_name] = {
                "completed": True,
                "step": i + 1,
            }

        state.status = "COMPLETED"
        state.progress = 100
        state.add_log("Pipeline complete.")

        if self.progress_callback:
            await self.progress_callback(state)

        yield state

    async def execute_single(self, state: MissionState, agent_index: int) -> MissionState:
        """Execute a single agent by index (for re-runs/corrections)."""
        if agent_index >= len(self.agents):
            state.set_error(f"Invalid agent index: {agent_index}")
            return state

        agent = self.agents[agent_index]
        state.add_log(f"Re-running agent: {agent.role_name}")
        state = await agent.run(state)
        return state
