from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any

from src.core.types import SkillCategory, SkillV2
from src.utils.logger import get_logger

logger = get_logger(__name__)


class SkillRegistry:
    def __init__(
        self,
        embed_fn: Callable[[str], Any],
        store_fn: Callable[[SkillV2], Any],
        search_fn: Callable[[list[float], int], Any] | None = None,
        llm_client: Any | None = None,
    ) -> None:
        self._embed_fn = embed_fn
        self._store_fn = store_fn
        self._search_fn = search_fn
        self._llm = llm_client

    async def register(self, skill: SkillV2) -> str:
        if not skill.embedding:
            text = f"{skill.name}: {skill.description} | tags: {', '.join(skill.tags)} | applicability: {skill.applicability}"
            skill.embedding = await self._embed_fn(text)
        skill_id = await self._store_fn(skill)
        logger.info("SkillRegistry: registered skill '{}' with id {}", skill.name, skill_id)
        return skill_id

    async def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        query_embedding = await self._embed_fn(query)
        if self._search_fn:
            results = await self._search_fn(query_embedding, top_k)
            return results
        logger.warning("SkillRegistry: no search_fn configured, returning empty")
        return []

    async def register_from_execution(
        self,
        task_description: str,
        actions: list[dict[str, Any]],
        outcome: str,
    ) -> str | None:
        if not self._llm:
            logger.warning("SkillRegistry: no llm_client for auto-registration")
            return None
        prompt = (
            f"Analyze this executed task and extract metadata for a reusable skill:\n\n"
            f"Task: {task_description}\n"
            f"Actions: {json.dumps(actions)[:500]}\n"
            f"Outcome: {outcome}\n\n"
            "Output JSON with: name, description, tags (array), applicability, boundaries, category.\n"
            "Example category: BUG_LOCALIZATION, PATCH_GENERATION, TEST_VERIFICATION, CODE_ANALYSIS."
        )
        response = await self._llm.chat([
            {"role": "system", "content": "You extract skill metadata from execution traces."},
            {"role": "user", "content": prompt},
        ])
        try:
            meta = json.loads(response)
            category_str = meta.get("category", "CODE_ANALYSIS")
            category = getattr(SkillCategory, category_str, SkillCategory.CODE_ANALYSIS)
            skill = SkillV2(
                name=meta.get("name", f"auto_{int(time.time())}"),
                category=category,
                description=meta.get("description", ""),
                tags=meta.get("tags", []),
                applicability=meta.get("applicability", ""),
                boundaries=meta.get("boundaries", ""),
                examples=[json.dumps({"task": task_description, "actions": actions, "outcome": outcome})],
            )
            skill_id = await self.register(skill)
            logger.info("SkillRegistry: auto-registered skill '{}' from execution", skill.name)
            return skill_id
        except (json.JSONDecodeError, KeyError) as e:
            logger.error("SkillRegistry: auto-registration failed: {}", e)
            return None
