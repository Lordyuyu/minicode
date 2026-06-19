"""
Real I/O integration tests for MiniCode agents.

These tests use actual file-system operations and subprocess calls
against a temporary codebase.  LLM calls remain mocked so the tests
are deterministic and do not require API credentials.
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.bug_locator import BugLocator
from src.agents.patch_generator import PatchGenerator
from src.core.state import AgentState
from src.core.types import BugReport, PatchResult


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_llm():
    """A DeepSeekClient mock that returns realistic JSON responses."""
    client = MagicMock()

    async def _chat(messages, **kwargs):
        # Return JSON with a plausible bug report
        return (
            '[{"file_path": "calculator.py", "line_start": 4, "line_end": 5, '
            '"error_type": "ZeroDivisionError", '
            '"error_message": "division by zero in divide()", '
            '"confidence": 0.95}]'
        )

    async def _chat_patch(messages, **kwargs):
        return (
            '{"file_path": "calculator.py", '
            '"original_content": "def divide(a, b):\\n    return a / b\\n", '
            '"patched_content": "def divide(a, b):\\n    if b == 0:\\n'
            '        raise ValueError(\'Cannot divide by zero\')\\n'
            '    return a / b\\n", '
            '"diff": "--- original\\n+++ patched\\n@@ -1,2 +1,4 @@\\n'
            ' def divide(a, b):\\n+    if b == 0:\\n'
            '+        raise ValueError(\'Cannot divide by zero\')\\n'
            '     return a / b\\n"}'
        )

    client.chat = AsyncMock(side_effect=_chat)
    client.chat_with_tools = AsyncMock(return_value=("ok", []))
    return client


@pytest.fixture
def agent_state(sample_codebase):
    """An AgentState wired to the temporary codebase."""
    return AgentState(
        task_id="test-real-001",
        input_codebase_path=sample_codebase,
        test_command="pytest",
    )


# ---------------------------------------------------------------------------
# BugLocator — real file I/O, mocked LLM
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_bug_locator_reads_real_files(mock_llm, agent_state):
    """BugLocator should read actual files from the sample codebase."""
    locator = BugLocator(mock_llm)

    # Simulate test output pointing at calculator.py
    agent_state.metadata["test_output"] = (
        "FAILED test_calculator.py::test_divide_by_zero - ZeroDivisionError\n"
        'File "sample_project/calculator.py", line 5, in divide\n'
    )

    reports = await locator.localize(agent_state)

    assert len(reports) >= 1
    bug = reports[0]
    assert bug.file_path == "calculator.py"
    assert bug.confidence > 0


@pytest.mark.asyncio
async def test_bug_locator_fallback_when_no_traceback_files(
    mock_llm, agent_state
):
    """When no .py files appear in the traceback, fall back to scanning."""
    locator = BugLocator(mock_llm)
    agent_state.metadata["test_output"] = "Some generic error without file paths"

    reports = await locator.localize(agent_state)

    # Should still produce at least one report from the fallback scan
    assert len(reports) >= 1


# ---------------------------------------------------------------------------
# PatchGenerator — real file I/O, mocked LLM
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_patch_generator_writes_to_disk(mock_llm, agent_state):
    """PatchGenerator should write the patched content to the actual file."""
    # Use the patch-specific mock
    mock_llm.chat = AsyncMock(
        return_value=(
            '{"file_path": "sample_project/calculator.py", '
            '"original_content": "def divide(a, b):\\n    return a / b\\n", '
            '"patched_content": "def divide(a, b):\\n    if b == 0:\\n'
            '        raise ValueError(\'Cannot divide by zero\')\\n'
            '    return a / b\\n", '
            '"diff": "--- original\\n+++ patched\\n"}'
        )
    )

    bug = BugReport(
        file_path="sample_project/calculator.py",
        line_start=1,
        line_end=2,
        error_type="ZeroDivisionError",
        error_message="division by zero",
        confidence=0.95,
    )
    agent_state.bug_reports = [bug]

    generator = PatchGenerator(mock_llm)
    patches = await generator.generate(agent_state)

    assert len(patches) == 1
    assert patches[0].file_path == "sample_project/calculator.py"
    assert "ValueError" in patches[0].patched_content
    assert len(patches[0].diff) > 0

    # Verify the file was actually written to disk
    file_path = os.path.join(
        agent_state.input_codebase_path, "sample_project/calculator.py"
    )
    assert os.path.isfile(file_path)
    with open(file_path, encoding="utf-8") as f:
        content = f.read()
    assert "ValueError" in content

    # Verify backup was created
    backup_path = file_path + ".minicode.bak"
    assert os.path.isfile(backup_path)
    with open(backup_path, encoding="utf-8") as f:
        original = f.read()
    assert "return a / b" in original
    assert "ValueError" not in original


@pytest.mark.asyncio
async def test_patch_generator_handles_unreadable_file(
    mock_llm, agent_state
):
    """PatchGenerator returns None for nonexistent files."""
    bug = BugReport(
        file_path="nonexistent.py",
        line_start=1,
        line_end=2,
        error_type="NameError",
        error_message="module not found",
        confidence=0.5,
    )
    agent_state.bug_reports = [bug]

    generator = PatchGenerator(mock_llm)
    patches = await generator.generate(agent_state)

    assert len(patches) == 0


@pytest.mark.asyncio
async def test_patch_generator_concurrent(mock_llm, agent_state):
    """Concurrent generation processes multiple bugs in parallel."""
    mock_llm.chat = AsyncMock(
        return_value=(
            '{"file_path": "sample_project/calculator.py", '
            '"original_content": "def add(a, b):\\n    return a + b\\n", '
            '"patched_content": "def add(a, b):\\n    return a + b\\n", '
            '"diff": ""}'
        )
    )

    bugs = [
        BugReport(
            file_path="sample_project/calculator.py",
            line_start=1, line_end=2,
            error_type="TestError",
            error_message="test failure",
            confidence=0.8,
        )
        for _ in range(3)
    ]
    agent_state.bug_reports = bugs

    generator = PatchGenerator(mock_llm)
    patches = await generator.generate_concurrent(agent_state)

    assert len(patches) == 3
    assert mock_llm.chat.call_count == 3
