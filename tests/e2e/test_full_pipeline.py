from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch

from src.core.state import AgentState
from src.core.types import PatchResult


@pytest.mark.asyncio
async def test_full_pipeline_with_mocks(sample_codebase: str):
    mock_patches = [
        PatchResult(
            file_path="sample_project/calculator.py",
            original_content="def divide(a, b):\n    return a / b\n",
            patched_content="def divide(a, b):\n    if b == 0:\n        return float('inf')\n    return a / b\n",
            diff="--- original\n+++ patched\n@@ -1 +1,3 @@\n-def divide(a, b):\n-    return a / b\n+def divide(a, b):\n+    if b == 0:\n+        return float('inf')\n+    return a / b\n",
            verified=True,
        )
    ]

    mock_state = AgentState(
        task_id="test-e2e-001",
        input_codebase_path=sample_codebase,
        test_command="pytest",
        bug_reports=[],
        patches=mock_patches,
        verification_results=[{"file_path": "sample_project/calculator.py", "verified": True}],
        pipeline_success=True,
    )

    with patch("src.main.MiniCodeGraph") as MockGraph:
        mock_instance = MockGraph.return_value
        mock_instance.run = AsyncMock(return_value=mock_state)

        from src.main import run_pipeline
        result = await run_pipeline(codebase_path=sample_codebase, test_command="pytest")

        assert result.pipeline_success is True
        assert len(result.patches) == 1
        assert result.patches[0].verified is True
        assert result.patches[0].file_path == "sample_project/calculator.py"


@pytest.mark.asyncio
async def test_pipeline_handles_no_bugs(sample_codebase: str):
    mock_state = AgentState(
        task_id="test-e2e-002",
        input_codebase_path=sample_codebase,
        test_command="pytest",
        bug_reports=[],
        patches=[],
        verification_results=[],
        pipeline_success=False,
    )

    with patch("src.main.MiniCodeGraph") as MockGraph:
        mock_instance = MockGraph.return_value
        mock_instance.run = AsyncMock(return_value=mock_state)

        from src.main import run_pipeline
        result = await run_pipeline(codebase_path=sample_codebase, test_command="pytest")

        assert result.pipeline_success is False
        assert len(result.patches) == 0
        assert len(result.verification_results) == 0
