from __future__ import annotations

import uuid
from dataclasses import fields

from langgraph.graph import END, StateGraph

from src.core.state import AgentState
from src.llm.deepseek_client import DeepSeekClient
from src.orchestration.nodes.execute import ExecutionNode
from src.orchestration.nodes.intent import IntentNode
from src.orchestration.nodes.plan import PlanningNode
from src.orchestration.nodes.verify import VerificationNode
from src.storage.redis_client import RedisClient
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Module-level client singletons — lazily initialized via _ensure_initialized()
_llm_client: DeepSeekClient | None = None
_redis_client: RedisClient | None = None
_initialized: bool = False


async def _ensure_initialized() -> None:
    """One-time async initialisation of shared infrastructure clients.

    Redis failure is non-fatal — the pipeline degrades gracefully without
    caching / chunk storage, logging a warning instead of crashing.
    """
    global _llm_client, _redis_client, _initialized
    if _initialized:
        return
    _llm_client = DeepSeekClient()
    _redis_client = RedisClient()
    try:
        await _redis_client.initialize()
    except Exception:
        logger.warning(
            "Redis initialization failed — pipeline will run without "
            "cache / chunk storage"
        )
    _initialized = True
    logger.info("MiniCode infrastructure initialized")


class MiniCodeGraph:
    def __init__(self) -> None:
        # Nodes are created eagerly but use lazily-initialized clients
        self._intent_node: IntentNode | None = None
        self._planning_node: PlanningNode | None = None
        self._execution_node: ExecutionNode | None = None
        self._verification_node: VerificationNode | None = None
        self._app: StateGraph | None = None

    @property
    def app(self) -> StateGraph:
        """Lazy accessor for the compiled LangGraph app.

        Initializes infrastructure and nodes on first access so that test
        code can inspect ``graph.app`` without needing a full ``run()``.
        """
        if self._app is None:
            import asyncio
            try:
                _ = asyncio.get_running_loop()
                # We're inside an async context — can't call sync init.
                # Return a partially-built graph for test inspection.
                self._build_nodes_sync()
            except RuntimeError:
                # No running loop — safe to call sync init
                self._build_nodes_sync()
            self._app = self._build_graph()
        return self._app

    def _build_nodes_sync(self) -> None:
        """Synchronous fallback for tests — creates nodes without Redis init."""
        global _llm_client, _redis_client
        if _llm_client is None:
            _llm_client = DeepSeekClient()
        if _redis_client is None:
            _redis_client = RedisClient()
        self._intent_node = IntentNode(_llm_client)
        self._planning_node = PlanningNode(_llm_client)
        self._execution_node = ExecutionNode(_llm_client, _redis_client)
        self._verification_node = VerificationNode(_llm_client)

    def _build_nodes(self) -> None:
        """Wire up nodes after infrastructure is ready."""
        assert _llm_client is not None and _redis_client is not None
        self._intent_node = IntentNode(_llm_client)
        self._planning_node = PlanningNode(_llm_client)
        self._execution_node = ExecutionNode(_llm_client, _redis_client)
        self._verification_node = VerificationNode(_llm_client)

    def _build_graph(self) -> StateGraph:
        assert self._intent_node is not None  # _build_nodes must be called first
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

    # -- Routing ----------------------------------------------------------

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

        # Reset errors at the start of each loop iteration so stale errors
        # from a previous iteration do not incorrectly terminate the loop.
        error_count = len(state.errors)
        if iteration > 1 and error_count > 0:
            state.metadata.setdefault("previous_errors", []).extend(state.errors)
            state.errors.clear()

        if iteration >= 3:
            logger.warning("Max iterations ({}) reached, ending.", iteration)
            return END
        if error_count > 3:
            logger.warning("Too many errors ({}), ending.", error_count)
            return END

        logger.info(
            "Pipeline iteration {} continuing, looping back to intent.", iteration
        )
        return "intent"

    # -- Run --------------------------------------------------------------

    async def run(
        self,
        input_codebase_path: str,
        test_command: str = "pytest",
        metadata: dict | None = None,
    ) -> AgentState:
        await _ensure_initialized()
        self._build_nodes()

        initial_state = AgentState(
            task_id=str(uuid.uuid4()),
            input_codebase_path=input_codebase_path,
            test_command=test_command,
            metadata=metadata or {},
        )

        app = self._build_graph()
        raw = await app.ainvoke(initial_state)

        # LangGraph may return a dict rather than the AgentState dataclass.
        if isinstance(raw, dict):
            valid_keys = {f.name for f in fields(AgentState)}
            final_state = AgentState(
                **{k: v for k, v in raw.items() if k in valid_keys}
            )
        else:
            final_state = raw

        logger.info("Graph execution complete for task {}", final_state.task_id)
        return final_state
