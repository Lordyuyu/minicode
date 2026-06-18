from __future__ import annotations

import time
from typing import Any

from src.core.types import MemoryEntry, MemoryType
from src.llm.deepseek_client import DeepSeekClient


class ProceduralMemory:
    def __init__(self, llm_client: DeepSeekClient) -> None:
        self._llm = llm_client

    async def extract_pattern(self, episodes: list[MemoryEntry]) -> str:
        episodes_text = "\n".join(
            f"Task: {e.task_description}\nActions: {e.actions}\nOutcome: {e.outcome}"
            for e in episodes
        )
        prompt = (
            "You are a pattern extraction assistant. Given the following episodic memory entries, "
            "extract a generalized step-by-step procedure that captures the reusable pattern.\n\n"
            f"{episodes_text}\n\n"
            "Provide a concise procedural pattern."
        )
        pattern = await self._llm.chat([
            {"role": "system", "content": "You extract reusable procedural patterns from experiences."},
            {"role": "user", "content": prompt},
        ])
        return pattern

    async def create_procedural_entry(
        self,
        task_description: str,
        pattern: str,
        embedding: list[float],
    ) -> MemoryEntry:
        entry = MemoryEntry(
            memory_type=MemoryType.PROCEDURAL,
            task_description=task_description,
            procedural_pattern=pattern,
            embedding=embedding,
            timestamp=time.time(),
        )
        return entry
