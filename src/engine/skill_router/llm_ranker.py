from __future__ import annotations

import re
from typing import Any

from src.llm.deepseek_client import DeepSeekClient
from src.llm.prompt_templates import build_skill_ranking_prompt
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LLMRanker:
    def __init__(self, llm_client: DeepSeekClient) -> None:
        self._llm = llm_client

    async def rank(
        self, task_description: str, candidates: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Re-rank the top-K vector candidates with an LLM.

        Returns the single best candidate dict.  When only one candidate
        exists it is returned directly (no LLM call).  When the LLM returns
        a response that cannot be parsed as a valid 1-based index, the
        first candidate is returned as a safe fallback.
        """
        if len(candidates) == 1:
            return candidates[0]

        if not candidates:
            logger.warning("LLMRanker called with empty candidate list")
            return {}

        prompt = build_skill_ranking_prompt(task_description, candidates)
        try:
            response = await self._llm.chat([
                {"role": "system", "content": "You select the best skill for the task."},
                {"role": "user", "content": prompt},
            ])
        except Exception:
            logger.exception("LLM ranking call failed, falling back to first candidate")
            return candidates[0]

        # Extract the first integer from the response (robust parsing).
        # The prompt asks for a single number, but LLMs often wrap it in text.
        match = re.search(r'\d+', response)
        if match is None:
            logger.warning(
                "LLM ranking response contained no number: {!r}, "
                "falling back to first candidate",
                response[:100],
            )
            return candidates[0]

        try:
            index = int(match.group()) - 1  # 1-based → 0-based
        except ValueError:
            logger.warning("Failed to parse ranking index, falling back to first candidate")
            return candidates[0]

        if index < 0 or index >= len(candidates):
            logger.warning(
                "LLM ranking index {} out of range [0, {}), "
                "falling back to first candidate",
                index + 1,
                len(candidates),
            )
            return candidates[0]

        return candidates[index]
