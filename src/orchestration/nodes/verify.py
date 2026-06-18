from __future__ import annotations

import subprocess
import sys
from src.core.state import AgentState
from src.core.types import AgentPhase
from src.llm.deepseek_client import DeepSeekClient
from src.utils.logger import get_logger

logger = get_logger(__name__)


class VerificationNode:
    def __init__(self, llm_client: DeepSeekClient) -> None:
        self._llm = llm_client

    async def execute(self, state: AgentState) -> AgentState:
        logger.info("Executing VerificationNode for task {}", state.task_id)
        if not state.patches:
            logger.warning("No patches to verify")
            state.pipeline_success = False
            state.current_phase = AgentPhase.INTENT_RECOGNITION
            return state

        all_verified = True
        for patch in state.patches:
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "pytest", state.input_codebase_path, "-x", "--tb=short"],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                patch.verified = result.returncode == 0
                patch.verification_output = result.stdout + "\n" + result.stderr[-500:]
                if not patch.verified:
                    all_verified = False
                    state.errors.append(f"Patch for {patch.file_path} failed: {result.stderr[:200]}")
                logger.info("Patch for {} verified: {}", patch.file_path, patch.verified)
            except subprocess.TimeoutExpired:
                patch.verified = False
                patch.verification_output = "Test timed out"
                all_verified = False
                state.errors.append(f"Verification timed out for {patch.file_path}")

        state.pipeline_success = all_verified
        state.current_phase = AgentPhase.INTENT_RECOGNITION
        logger.info("Pipeline success: {}", state.pipeline_success)
        return state
