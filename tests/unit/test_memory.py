from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from src.core.types import MemoryEntry, MemoryType
from src.engine.memory.episodic import EpisodicMemory
from src.engine.memory.procedural import ProceduralMemory
from src.engine.memory.consolidator import MemoryConsolidator


@pytest.mark.asyncio
async def test_episodic_record():
    store_fn = AsyncMock()
    memory = EpisodicMemory(store_fn)

    entry = await memory.record(
        task_id="task-1",
        description="Test task",
        actions=[{"action": "read_file", "path": "test.py"}],
        outcome="success",
    )

    assert entry.memory_type == MemoryType.EPISODIC
    assert entry.task_description == "Test task"
    assert entry.actions == [{"action": "read_file", "path": "test.py"}]
    assert entry.outcome == "success"
    store_fn.assert_awaited_once_with(entry)


@pytest.mark.asyncio
async def test_procedural_extract_pattern():
    llm = MagicMock()
    llm.chat = AsyncMock(return_value="1. Analyze the code\n2. Identify the bug\n3. Generate a patch")
    memory = ProceduralMemory(llm)

    episodes = [
        MemoryEntry(
            memory_type=MemoryType.EPISODIC,
            task_description="Fix division by zero",
            actions=[{"action": "patch", "file": "calc.py"}],
            outcome="Tests passed",
        ),
        MemoryEntry(
            memory_type=MemoryType.EPISODIC,
            task_description="Fix null pointer",
            actions=[{"action": "patch", "file": "main.py"}],
            outcome="All checks green",
        ),
    ]

    pattern = await memory.extract_pattern(episodes)
    assert "Analyze" in pattern
    assert "Identify" in pattern
    llm.chat.assert_awaited_once()


@pytest.mark.asyncio
async def test_consolidator_offline_reflection():
    embed_fn = AsyncMock(return_value=[0.1, 0.2, 0.3])
    search_fn = AsyncMock(
        return_value=[
            {
                "task_description": "Previous fix",
                "outcome": "resolved",
                "timestamp": 1000.0,
            }
        ]
    )
    store_fn = AsyncMock()
    llm = MagicMock()
    llm.chat = AsyncMock(return_value="1. Reproduce\n2. Fix\n3. Verify")

    consolidator = MemoryConsolidator(
        llm_client=llm,
        store_fn=store_fn,
        search_fn=search_fn,
        embed_fn=embed_fn,
    )

    entry = await consolidator.reflect(
        task_id="task-2",
        task_description="Fix database connection timeout",
        outcome="connection pool increased",
    )

    assert entry is not None
    assert isinstance(entry, MemoryEntry)
    store_fn.assert_awaited_once()
