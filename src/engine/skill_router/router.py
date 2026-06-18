from __future__ import annotations

from typing import Any, Callable

from src.core.types import Skill
from src.engine.skill_router.llm_ranker import LLMRanker
from src.engine.skill_router.vector_ranker import VectorRanker


class SkillRouter:
    def __init__(
        self, embed_fn: Callable[..., Any], llm_client: Any, vector_top_k: int = 10
    ) -> None:
        self._vector_ranker = VectorRanker(embed_fn)
        self._llm_ranker = LLMRanker(llm_client)
        self._vector_top_k = vector_top_k

    async def route(
        self, task_description: str, skill_pool: list[Skill]
    ) -> Skill | None:
        if not skill_pool:
            return None
        candidates = await self._vector_ranker.rank(
            task_description, skill_pool, top_k=self._vector_top_k
        )
        if not candidates:
            return None
        best = await self._llm_ranker.rank(task_description, candidates)
        return best["skill"]
