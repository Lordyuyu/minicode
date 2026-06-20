from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from src.core.types import SkillCategory, SkillV2
from src.engine.skill_router.registry import SkillRegistry


@pytest.mark.asyncio
async def test_registry_register_skill():
    embed_fn = AsyncMock(return_value=[0.1] * 768)
    store_fn = AsyncMock()
    registry = SkillRegistry(embed_fn, store_fn)
    skill = SkillV2(name="test_skill", category=SkillCategory.CODE_ANALYSIS, description="test")
    skill_id = await registry.register(skill)
    assert skill_id is not None
    store_fn.assert_called_once()
    embed_fn.assert_called_once()


@pytest.mark.asyncio
async def test_registry_search_with_metadata():
    embed_fn = AsyncMock(return_value=[0.1] * 768)
    store_fn = AsyncMock()
    search_fn = AsyncMock(return_value=[
        {
            "name": "bug_localizer",
            "category": "BUG_LOCALIZATION",
            "similarity": 0.9,
            "tags": ["bug", "error"],
            "applicability": "test failures",
        },
    ])
    registry = SkillRegistry(embed_fn, store_fn, search_fn)
    results = await registry.search("find bugs in test output", top_k=5)
    assert len(results) == 1
    assert results[0]["name"] == "bug_localizer"
    assert results[0]["similarity"] == 0.9
    embed_fn.assert_called_once_with("find bugs in test output")
    search_fn.assert_called_once_with([0.1] * 768, 5)


@pytest.mark.asyncio
async def test_registry_post_execution_register():
    embed_fn = AsyncMock(return_value=[0.1] * 768)
    store_fn = AsyncMock()
    llm = AsyncMock()
    llm.chat = AsyncMock(return_value=json.dumps({
        "name": "auto_bug_fixer",
        "description": "Fixes division by zero errors",
        "tags": ["bug", "division", "zero"],
        "applicability": "When a ZeroDivisionError occurs",
        "boundaries": "Only for arithmetic operations",
        "category": "BUG_LOCALIZATION",
    }))
    registry = SkillRegistry(embed_fn, store_fn, llm_client=llm)
    skill_id = await registry.register_from_execution(
        task_description="fix division by zero in calculator.py",
        actions=[{"tool": "read_file", "path": "calculator.py"}],
        outcome="success",
    )
    assert skill_id is not None
    embed_fn.assert_called_once()
    store_fn.assert_called_once()
