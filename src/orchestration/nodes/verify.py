from __future__ import annotations

import asyncio
import sys

from src.core.state import AgentState
from src.core.types import AgentPhase
from src.llm.deepseek_client import DeepSeekClient
from src.utils.logger import get_logger

logger = get_logger(__name__)

_VERIFY_TIMEOUT = 60  # seconds per test run


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

        # Run pytest ONCE for the fully-patched codebase (all patches have
        # already been applied in-place by PatchGenerator).
        success, output = await self._run_pytest_once(state)

        all_verified = success
        for patch in state.patches:
            patch.verified = success
            patch.verification_output = output
            if not success:
                state.errors.append(
                    f"Patch for {patch.file_path} failed verification"
                )
            logger.info(
                "Patch for {} verified: {}", patch.file_path, patch.verified
            )

        state.pipeline_success = all_verified
        state.current_phase = AgentPhase.INTENT_RECOGNITION
        logger.info("Pipeline success: {}", state.pipeline_success)
        return state

    async def _run_pytest_once(self, state: AgentState) -> tuple[bool, str]:
        """Run the test suite ONCE using ``asyncio.create_subprocess_exec``.

        Returns ``(success: bool, output: str)``.
        """
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m", "pytest",
                state.input_codebase_path,
                "-x", "--tb=short",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=_VERIFY_TIMEOUT,
            )
            stdout = stdout_bytes.decode("utf-8", errors="replace") if stdout_bytes else ""
            stderr = stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else ""
            output = stdout + "\n" + stderr
            return proc.returncode == 0, output
        except TimeoutError:
            return False, f"Test timed out after {_VERIFY_TIMEOUT}s"
        except Exception as exc:
            logger.exception("Pytest execution failed")
            return False, f"Failed to run tests: {exc}"
