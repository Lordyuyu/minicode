from __future__ import annotations

from src.core.state import AgentState
from src.core.types import AgentPhase
from src.llm.deepseek_client import DeepSeekClient
from src.llm.prompt_templates import build_intent_recognition_prompt
from src.utils.logger import get_logger

logger = get_logger(__name__)


class IntentNode:
    def __init__(self, llm_client: DeepSeekClient) -> None:
        self._llm = llm_client

    async def execute(self, state: AgentState) -> AgentState:
        logger.info("Executing IntentNode for task {}", state.task_id)
        test_output = state.metadata.get("test_output", "No test output available")
        messages = build_intent_recognition_prompt(state.input_codebase_path, test_output)
        try:
            response = await self._llm.chat(messages)
            import json
            result = json.loads(response)
            state.identified_intent = result.get("intent", "bug_fix")
            state.metadata["intent_confidence"] = result.get("confidence", 0.0)
            state.metadata["intent_reasoning"] = result.get("reasoning", "")
            logger.info("Identified intent: {} (confidence: {})", state.identified_intent, state.metadata["intent_confidence"])
        except Exception as e:
            logger.error("Intent recognition failed: {}", e)
            state.identified_intent = "bug_fix"
        state.current_phase = AgentPhase.PLANNING
        return state
