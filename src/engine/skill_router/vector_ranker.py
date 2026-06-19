from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.core.types import Skill
from src.utils.embedding import cosine_similarity


class VectorRanker:
    def __init__(self, embed_fn: Callable[..., Any]) -> None:
        self._embed_fn = embed_fn

    async def rank(
        self, query: str, skill_pool: list[Skill], top_k: int = 10
    ) -> list[dict[str, Any]]:
        query_embedding = await self._embed_fn(query)
        scored = []
        for skill in skill_pool:
            sim = cosine_similarity(query_embedding, skill.embedding)
            scored.append({
                "skill": skill,
                "name": skill.name,
                "category": skill.category.value if hasattr(skill.category, "value") else str(skill.category),
                "description": skill.description,
                "similarity": sim,
            })
        scored.sort(key=lambda x: x["similarity"], reverse=True)
        return scored[:top_k]
