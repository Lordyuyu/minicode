from __future__ import annotations

import asyncio
import json
from typing import Any

from src.core.state import AgentState
from src.core.types import PermissionScope, SubAgentTask
from src.engine.orchestrator.permission import PermissionController
from src.llm.deepseek_client import DeepSeekClient
from src.utils.logger import get_logger
from src.utils.text import extract_json

logger = get_logger(__name__)

_DECOMPOSE_PROMPT = """You are a task decomposition engine.
Given the current agent state, break down the work into parallel sub-agent tasks.

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
        """Decompose the current state into sub-agent tasks via LLM."""
        state_summary = json.dumps(
            {
                "task_id": state.task_id,
                "intent": state.identified_intent,
                "bugs": [
                    {
                        "file_path": b.file_path,
                        "line_start": b.line_start,
                        "line_end": b.line_end,
                        "error_type": b.error_type,
                        "error_message": b.error_message,
                        "confidence": b.confidence,
                    }
                    for b in state.bug_reports
                ],
                "phase": state.current_phase.value,
            },
            indent=2,
        )

        prompt = _DECOMPOSE_PROMPT.format(state_summary=state_summary)
        response = await self._llm.chat([{"role": "user", "content": prompt}])

        tasks = self._parse_tasks(response)
        logger.info(
            "Dispatched {} sub-agent tasks from state {}", len(tasks), state.trace_id
        )
        return tasks

    async def dispatch_concurrent(
        self,
        state: AgentState,
        tasks: list[SubAgentTask],
        executor_fn: Any = None,
    ) -> list[SubAgentTask]:
        """Execute *tasks* concurrently via ``asyncio.gather``.

        Tasks with no cross-dependencies are run in parallel.  Tasks that
        depend on another task's completion are deferred until all
        dependencies resolve.

        Args:
            state: Current agent state (shared read-only for metadata).
            tasks: Sub-agent tasks to execute.
            executor_fn: Async callable ``(state, task) -> SubAgentTask``.
                If ``None``, tasks are left as-is (for decomposition-only
                use cases).

        Returns:
            The completed task list with ``result`` and ``success`` populated.
        """
        if not executor_fn:
            return tasks

        # Resolve dependencies: group tasks into waves.
        # Wave 0 = tasks with no unsatisfied dependencies.
        completed: dict[str, SubAgentTask] = {}
        remaining = list(tasks)

        while remaining:
            wave: list[SubAgentTask] = []
            deferred: list[SubAgentTask] = []
            for t in remaining:
                deps_satisfied = all(
                    d in completed for d in t.dependencies
                )
                if deps_satisfied:
                    wave.append(t)
                else:
                    deferred.append(t)

            if not wave:
                logger.error(
                    "Unresolvable dependencies among tasks: {}",
                    [t.agent_type for t in remaining],
                )
                break

            logger.info(
                "Dispatching wave of {} concurrent sub-agent(s)", len(wave)
            )
            results = await asyncio.gather(
                *(executor_fn(state, t) for t in wave),
                return_exceptions=True,
            )

            for task, result in zip(wave, results):
                if isinstance(result, Exception):
                    task.success = False
                    task.result = f"Sub-agent failed: {result}"
                    logger.error(
                        "Sub-agent {} ({}) failed: {}",
                        task.agent_id,
                        task.agent_type,
                        result,
                    )
                else:
                    task.success = result.success
                    task.result = result.result
                completed[task.agent_id] = task

            remaining = deferred

        return tasks

    async def aggregate_results(
        self, state: AgentState, sub_results: list[SubAgentTask]
    ) -> AgentState:
        """Merge sub-agent results into the shared state.

        Includes conflict detection: if multiple agents modified the same
        file, a warning is logged and the conflict is recorded in metadata.
        """
        state.metadata["sub_agent_results"] = [
            {
                "agent_id": t.agent_id,
                "agent_type": t.agent_type,
                "task_description": t.task_description,
                "permission": t.permission.value,
                "success": t.success,
                "result": t.result,
            }
            for t in sub_results
        ]

        # Conflict detection — flag overlapping target files
        file_owners: dict[str, list[str]] = {}
        for t in sub_results:
            if not t.success:
                continue
            for f in t.target_files:
                file_owners.setdefault(f, []).append(t.agent_type)
        conflicts = {
            f: owners for f, owners in file_owners.items() if len(owners) > 1
        }
        if conflicts:
            logger.warning(
                "File conflict detected — multiple agents touched the same file(s): {}",
                conflicts,
            )
            state.metadata["file_conflicts"] = conflicts

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
        """Parse LLM JSON response into ``SubAgentTask`` objects.

        Uses the shared ``extract_json`` utility for robust fence-stripping.
        """
        cleaned = extract_json(response)

        try:
            raw = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.error(
                "Failed to parse LLM response as JSON: {}", response[:200]
            )
            return []

        if isinstance(raw, dict):
            raw = [raw]

        tasks: list[SubAgentTask] = []
        for item in raw:
            perm_str = item.get("permission", "read_only").upper()
            try:
                perm = PermissionScope(perm_str)
            except ValueError:
                perm = PermissionScope.READ_ONLY

            task = SubAgentTask(
                agent_type=item.get("agent_type", "unknown"),
                task_description=item.get("task_description", ""),
                permission=perm,
                target_files=item.get("target_files", []),
                dependencies=item.get("dependencies", []),
            )
            tasks.append(task)

        return tasks
