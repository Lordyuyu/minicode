from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from src.core.types import Skill, SkillCategory
from src.engine.skill_router.llm_ranker import LLMRanker
from src.engine.skill_router.router import SkillRouter
from src.engine.skill_router.vector_ranker import VectorRanker


@pytest.mark.asyncio
async def test_vector_ranker_rank():
    embed_fn = AsyncMock(return_value=[1.0, 0.0, 0.0])
    ranker = VectorRanker(embed_fn)
    skill1 = Skill(
        name="bug_finder",
        category=SkillCategory.BUG_LOCALIZATION,
        description="Finds bugs in code",
        embedding=[1.0, 0.0, 0.0],
    )
    skill2 = Skill(
        name="patch_gen",
        category=SkillCategory.PATCH_GENERATION,
        description="Generates patches",
        embedding=[0.0, 1.0, 0.0],
    )
    results = await ranker.rank("find bugs", [skill1, skill2], top_k=2)

    assert len(results) == 2
    assert all("similarity" in r for r in results)
    assert results[0]["similarity"] > results[1]["similarity"]
    assert results[0]["name"] == "bug_finder"


@pytest.mark.asyncio
async def test_llm_ranker_rank():
    llm = AsyncMock()
    llm.chat = AsyncMock(return_value="1")
    ranker = LLMRanker(llm)
    candidates = [
        {"name": "bug_finder", "category": "BUG_LOCALIZATION", "description": "Finds bugs", "similarity": 0.9},
        {"name": "patch_gen", "category": "PATCH_GENERATION", "description": "Generates patches", "similarity": 0.5},
    ]
    result = await ranker.rank("find bugs", candidates)

    assert result["name"] == "bug_finder"
    llm.chat.assert_called_once()


@pytest.mark.asyncio
async def test_skill_router_full_pipeline():
    embed_fn = AsyncMock(return_value=[1.0, 0.0, 0.0])
    llm = AsyncMock()
    llm.chat = AsyncMock(return_value="1")
    router = SkillRouter(embed_fn, llm)

    skill1 = Skill(
        name="bug_finder",
        category=SkillCategory.BUG_LOCALIZATION,
        description="Finds bugs in code",
        embedding=[1.0, 0.0, 0.0],
    )
    skill2 = Skill(
        name="patch_gen",
        category=SkillCategory.PATCH_GENERATION,
        description="Generates patches",
        embedding=[0.0, 1.0, 0.0],
    )
    result = await router.route("find bugs", [skill1, skill2])

    assert result is not None
    assert result.name == "bug_finder"
    embed_fn.assert_called_once_with("find bugs")
    llm.chat.assert_called_once()
