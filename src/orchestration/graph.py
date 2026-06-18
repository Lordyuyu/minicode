from __future__ import annotations

from langgraph.graph import StateGraph, END
from src.core.state import AgentState
from src.core.types import AgentPhase
from src.llm.deepseek_client import DeepSeekClient
from src.storage.redis_client import RedisClient
from src.orchestration.nodes.intent import IntentNode
from src.orchestration.nodes.plan import PlanningNode
from src.orchestration.nodes.execute import ExecutionNode
from src.orchestration.nodes.verify import VerificationNode
from src.utils.logger import get_logger

logger = get_logger(__name__)

_llm_client = DeepSeekClient()
_redis_client = RedisClient()


class MiniCodeGraph:
    def __init__(self) -> None:
        self._intent_node = IntentNode(_llm_client)
        self._planning_node = PlanningNode(_llm_client)
        self._execution_node = ExecutionNode(_llm_client, _redis_client)
        self._verification_node = VerificationNode(_llm_client)
        self.app = self._build_graph()

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(AgentState)

        workflow.add_node("intent", self._intent_node.execute)
        workflow.add_node("plan", self._planning_node.execute)
        workflow.add_node("execute", self._execution_node.execute)
        workflow.add_node("verify", self._verification_node.execute)

        workflow.add_conditional_edges(
            "intent",
            self._route_from_intent,
            {"plan": "plan", END: END},
        )
        workflow.add_conditional_edges(
            "plan",
            self._route_from_plan,
            {"execute": "execute", END: END},
        )
        workflow.add_edge("execute", "verify")
        workflow.add_conditional_edges(
            "verify",
            self._route_from_verify,
            {"intent": "intent", END: END},
        )

        workflow.set_entry_point("intent")
        return workflow.compile()

    @staticmethod
    def _route_from_intent(state: AgentState) -> str:
        return "plan" if state.identified_intent else END

    @staticmethod
    def _route_from_plan(state: AgentState) -> str:
        return "execute" if state.execution_plan else END

    @staticmethod
    def _route_from_verify(state: AgentState) -> str:
        if state.pipeline_success:
            logger.info("Pipeline succeeded, ending.")
            return END
        iteration = state.metadata.get("loop_iteration", 0) + 1
        state.metadata["loop_iteration"] = iteration
        if iteration >= 3 or len(state.errors) > 3:
            logger.warning("Max iterations ({}) or too many errors ({}), ending.", iteration, len(state.errors))
            return END
        logger.info("Pipeline iteration {} continuing, looping back to intent.", iteration)
        return "intent"

    async def run(
        self,
        input_codebase_path: str,
        test_command: str = "pytest",
        metadata: dict | None = None,
    ) -> AgentState:
        import uuid
        initial_state = AgentState(
            task_id=str(uuid.uuid4()),
            input_codebase_path=input_codebase_path,
            test_command=test_command,
            metadata=metadata or {},
        )
        raw = await self.app.ainvoke(initial_state)
        if isinstance(raw, dict):
            valid_keys = {f.name for f in __import__('dataclasses').fields(AgentState)}
            final_state = AgentState(**{k: v for k, v in raw.items() if k in valid_keys})
        else:
            final_state = raw
        logger.info("Graph execution complete for task {}", final_state.task_id)
        return final_state
