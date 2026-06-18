from __future__ import annotations

import json
from typing import Any

from src.core.state import AgentState
from src.core.types import SubAgentTask
from src.llm.deepseek_client import DeepSeekClient
from src.engine.orchestrator.permission import PermissionController
from src.utils.logger import get_logger

logger = get_logger(__name__)

_DECOMPOSE_PROMPT = """You are a task decomposition engine. Given the current agent state, break down the work into parallel sub-agent tasks.

Return a JSON array of tasks, where each task has:
- agent_type: type of sub-agent to dispatch
- task_description: clear description of what this sub-agent should do
- permission: one of "READ_ONLY", "CODE_WRITE", "TEST_EXEC", or "FULL_ACCESS"
- target_files: list of file paths this agent may need
- dependencies: list of agent_type values that must complete before this task begins

Current state summary:
{state_summary}

Return ONLY valid JSON, no markdown or explanation."""


class OrchestratorAgent:
    def __init__(
        self,
        llm_client: DeepSeekClient,
        permission_controller: PermissionController | None = None,
    ) -> None:
        self._llm = llm_client
        self._permission = permission_controller or PermissionController()

    async def plan_and_dispatch(self, state: AgentState) -> list[SubAgentTask]:
        state_summary = json.dumps({
            "task_id": state.task_id,
            "intent": state.identified_intent,
            "files": state.bug_reports,
            "phase": state.current_phase.value,
        }, indent=2)

        prompt = _DECOMPOSE_PROMPT.format(state_summary=state_summary)
        response = await self._llm.chat([{"role": "user", "content": prompt}])

        tasks = self._parse_tasks(response)
        logger.info("Dispatched {} sub-agent tasks from state {}", len(tasks), state.trace_id)
        return tasks

    async def aggregate_results(
        self, state: AgentState, sub_results: list[SubAgentTask]
    ) -> AgentState:
        state.metadata["sub_agent_results"] = [
            {
                "agent_id": t.agent_id,
                "agent_type": t.agent_type,
                "task_description": t.task_description,
                "success": t.success,
                "result": t.result,
            }
            for t in sub_results
        ]

        all_success = all(t.success for t in sub_results)
        combined = "\n".join(t.result for t in sub_results if t.result)

        state.metadata["sub_agents_success"] = all_success
        state.metadata["sub_agents_combined"] = combined

        logger.info(
            "Aggregated {} sub-agent results (all_success={})",
            len(sub_results),
            all_success,
        )
        return state

    def _parse_tasks(self, response: str) -> list[SubAgentTask]:
        cleaned = response.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            cleaned = "\n".join(lines[1:-1])

        try:
            raw = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.error("Failed to parse LLM response as JSON: {}", response[:200])
            return []

        if isinstance(raw, dict):
            raw = [raw]

        tasks: list[SubAgentTask] = []
        for item in raw:
            task = SubAgentTask(
                agent_type=item.get("agent_type", "unknown"),
                task_description=item.get("task_description", ""),
                target_files=item.get("target_files", []),
                dependencies=item.get("dependencies", []),
            )
            tasks.append(task)

        return tasks
