from __future__ import annotations

import json

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.core.state import AgentState
from src.core.types import PermissionScope, SubAgentTask
from src.engine.orchestrator.permission import PermissionController
from src.engine.orchestrator.coordinator import OrchestratorAgent


@pytest.mark.asyncio
async def test_permission_controller_grants_access():
    controller = PermissionController()

    assert controller.check_access(PermissionScope.READ_ONLY, "read_file") is True
    assert controller.check_access(PermissionScope.READ_ONLY, "write_file") is False
    assert controller.check_access(PermissionScope.CODE_WRITE, "read_file") is True
    assert controller.check_access(PermissionScope.CODE_WRITE, "write_file") is True
    assert controller.check_access(PermissionScope.CODE_WRITE, "run_test") is False
    assert controller.check_access(PermissionScope.TEST_EXEC, "run_test") is True
    assert controller.check_access(PermissionScope.FULL_ACCESS, "shell_exec") is True

    ok, msg = controller.validate_tool_call("read_file", {"path": "/tmp/test.py"}, PermissionScope.READ_ONLY)
    assert ok is True
    assert msg == ""

    ok, msg = controller.validate_tool_call("write_file", {"path": "/tmp/test.py"}, PermissionScope.READ_ONLY)
    assert ok is False
    assert "not allowed" in msg


@pytest.mark.asyncio
async def test_orchestrator_dispatch():
    llm = MagicMock()
    llm.chat = AsyncMock(return_value=json_response())
    controller = MagicMock()
    orchestrator = OrchestratorAgent(llm_client=llm, permission_controller=controller)

    state = AgentState(
        task_id="task-001",
        trace_id="trace-001",
        identified_intent="fix division by zero bug",
    )

    tasks = await orchestrator.plan_and_dispatch(state)

    assert len(tasks) == 2
    assert tasks[0].agent_type == "bug_locator"
    assert tasks[1].agent_type == "patch_generator"
    assert tasks[0].task_description != ""
    assert tasks[1].task_description != ""


@pytest.mark.asyncio
async def test_orchestrator_aggregate():
    llm = MagicMock()
    orchestrator = OrchestratorAgent(llm_client=llm)

    state = AgentState(task_id="task-002", trace_id="trace-002")
    sub_results = [
        SubAgentTask(
            agent_id="a1",
            agent_type="bug_locator",
            task_description="locate the bug",
            result="Found bug in calculator.py:17",
            success=True,
        ),
        SubAgentTask(
            agent_id="a2",
            agent_type="patch_generator",
            task_description="generate patch",
            result="Patch applied to calculator.py",
            success=False,
        ),
    ]

    updated = await orchestrator.aggregate_results(state, sub_results)

    assert updated.metadata["sub_agents_success"] is False
    assert len(updated.metadata["sub_agent_results"]) == 2
    assert "Found bug in calculator.py:17" in updated.metadata["sub_agents_combined"]


def json_response() -> str:
    return json.dumps([
        {
            "agent_type": "bug_locator",
            "task_description": "Identify the exact file and line of the division by zero",
            "permission": "READ_ONLY",
            "target_files": ["calculator.py"],
            "dependencies": [],
        },
        {
            "agent_type": "patch_generator",
            "task_description": "Generate a patch to handle division by zero",
            "permission": "CODE_WRITE",
            "target_files": ["calculator.py"],
            "dependencies": ["bug_locator"],
        },
    ])
