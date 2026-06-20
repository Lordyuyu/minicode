from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from src.core.types import MemoryEntry, MemoryType
from src.storage.repository.memory_repo import MemoryRepository
from src.storage.repository.skill_repo import SkillRepository
from src.storage.repository.task_repo import TaskRepository


@pytest.mark.asyncio
async def test_task_repo_create_and_get():
    repo = TaskRepository(AsyncMock())
    repo.session.execute = AsyncMock()
    repo.session.commit = AsyncMock()
    result = await repo.create_task("/test/path", "pytest -x")
    assert result is not None
    repo.session.execute.assert_called_once()


@pytest.mark.asyncio
async def test_skill_repo_vector_search():
    repo = SkillRepository(AsyncMock())
    repo.session.execute = AsyncMock()
    repo.session.execute.return_value = AsyncMock()
    repo.session.execute.return_value.fetchall = AsyncMock(return_value=[])
    results = await repo.search_similar([0.1] * 768, top_k=5)
    assert isinstance(results, list)


@pytest.mark.asyncio
async def test_memory_repo_store_entry():
    repo = MemoryRepository(AsyncMock())
    repo.session.execute = AsyncMock()
    repo.session.commit = AsyncMock()
    entry = MemoryEntry(memory_type=MemoryType.PROCEDURAL, task_description="test")
    await repo.store(entry)
    repo.session.execute.assert_called_once()
