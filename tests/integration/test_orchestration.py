from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from src.core.state import AgentState
from src.core.types import AgentPhase
from src.orchestration.graph import MiniCodeGraph
from src.orchestration.nodes.execute import ExecutionNode
from src.orchestration.nodes.intent import IntentNode
from src.orchestration.nodes.plan import PlanningNode
from src.orchestration.nodes.verify import VerificationNode


@pytest.mark.asyncio
async def test_intent_node():
    node = IntentNode(AsyncMock())
    state = AgentState(
        task_id="test-1",
        input_codebase_path="/fake/path",
        test_command="pytest",
    )
    result = await node.execute(state)
    assert result.current_phase == AgentPhase.PLANNING


@pytest.mark.asyncio
async def test_planning_node():
    node = PlanningNode(AsyncMock())
    state = AgentState(
        task_id="test-1",
        identified_intent="bug_fix",
    )
    result = await node.execute(state)
    assert result.current_phase == AgentPhase.EXECUTION
    assert len(result.execution_plan) > 0


@pytest.mark.asyncio
async def test_execution_node():
    node = ExecutionNode(AsyncMock(), AsyncMock())
    state = AgentState(
        task_id="test-1",
        execution_plan=["step1", "step2"],
    )
    result = await node.execute(state)
    assert result.current_phase == AgentPhase.VERIFICATION


@pytest.mark.asyncio
async def test_verification_node():
    node = VerificationNode(AsyncMock())
    state = AgentState(
        task_id="test-1",
        test_command="pytest",
        input_codebase_path="/fake/path",
    )
    result = await node.execute(state)
    assert result.current_phase == AgentPhase.INTENT_RECOGNITION or result.pipeline_success is not None


@pytest.mark.asyncio
async def test_full_graph_creation():
    graph = MiniCodeGraph()
    assert graph.app is not None
