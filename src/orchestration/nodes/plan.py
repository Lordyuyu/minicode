from __future__ import annotations

from src.core.state import AgentState
from src.core.types import AgentPhase
from src.llm.deepseek_client import DeepSeekClient
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PlanningNode:
    def __init__(self, llm_client: DeepSeekClient) -> None:
        self._llm = llm_client

    async def execute(self, state: AgentState) -> AgentState:
        logger.info("Executing PlanningNode for intent: {}", state.identified_intent)
        intent = state.identified_intent
        if intent == "bug_fix":
            state.execution_plan = [
                "run_tests_to_get_errors",
                "localize_bugs",
                "generate_patches",
                "verify_patches",
            ]
        elif intent == "feature_add":
            state.execution_plan = [
                "analyze_codebase",
                "design_feature",
                "implement_feature",
                "run_tests",
            ]
        elif intent == "refactor":
            state.execution_plan = [
                "analyze_codebase",
                "plan_refactoring",
                "apply_refactoring",
                "run_tests",
            ]
        else:
            state.execution_plan = [
                "run_tests",
                "analyze_output",
                "determine_action",
            ]
        state.current_step_index = 0
        state.current_phase = AgentPhase.EXECUTION
        logger.info("Execution plan: {}", state.execution_plan)
        return state
