from __future__ import annotations

import asyncio
from src.core.state import AgentState
from src.core.types import AgentPhase
from src.llm.deepseek_client import DeepSeekClient
from src.storage.redis_client import RedisClient
from src.engine.skill_router.router import SkillRouter
from src.engine.context_compressor.hierarchical import HierarchicalCompressor
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ExecutionNode:
    def __init__(
        self,
        llm_client: DeepSeekClient,
        redis_client: RedisClient,
        skill_router: SkillRouter | None = None,
        compressor: HierarchicalCompressor | None = None,
    ) -> None:
        self._llm = llm_client
        self._redis = redis_client
        self._skill_router = skill_router
        self._compressor = compressor

    async def execute(self, state: AgentState) -> AgentState:
        logger.info("Executing ExecutionNode, plan: {}", state.execution_plan)
        for step in state.execution_plan:
            logger.info("Processing step: {}", step)
            if step == "run_tests_to_get_errors":
                state.metadata["test_output"] = await self._run_tests(state)
            elif step == "localize_bugs":
                from src.agents.bug_locator import BugLocator
                locator = BugLocator(self._llm)
                state.bug_reports = await locator.localize(state)
            elif step == "generate_patches":
                from src.agents.patch_generator import PatchGenerator
                generator = PatchGenerator(self._llm)
                state.patches = await generator.generate(state)
        state.current_phase = AgentPhase.VERIFICATION
        return state

    async def _run_tests(self, state: AgentState) -> str:
        import subprocess
        import sys
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", state.input_codebase_path, "-x", "--tb=short"],
                capture_output=True,
                text=True,
                timeout=60,
            )
            return result.stdout + "\n" + result.stderr
        except subprocess.TimeoutExpired as e:
            return f"Test timed out after 60s\nstdout: {e.stdout}\nstderr: {e.stderr}"
        except Exception as e:
            return f"Failed to run tests: {e}"
