from __future__ import annotations

from typing import Any

from src.llm.deepseek_client import DeepSeekClient
from src.llm.prompt_templates import build_skill_ranking_prompt


class LLMRanker:
    def __init__(self, llm_client: DeepSeekClient) -> None:
        self._llm = llm_client

    async def rank(
        self, task_description: str, candidates: list[dict[str, Any]]
    ) -> dict[str, Any]:
        if len(candidates) == 1:
            return candidates[0]
        prompt = build_skill_ranking_prompt(task_description, candidates)
        response = await self._llm.chat([
            {"role": "system", "content": "You select the best skill for the task."},
            {"role": "user", "content": prompt},
        ])
        index = int(response.strip()) - 1
        return candidates[index]
