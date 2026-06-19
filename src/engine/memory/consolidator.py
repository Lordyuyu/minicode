from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from src.core.types import MemoryEntry, MemoryType
from src.engine.memory.episodic import EpisodicMemory
from src.engine.memory.procedural import ProceduralMemory
from src.llm.deepseek_client import DeepSeekClient


class MemoryConsolidator:
    def __init__(
        self,
        llm_client: DeepSeekClient,
        store_fn: Callable[[MemoryEntry], Any],
        search_fn: Callable[[list[float], int], list[dict[str, Any]]],
        embed_fn: Callable[[str], list[float]],
    ) -> None:
        self._llm = llm_client
        self._store_fn = store_fn
        self._search_fn = search_fn
        self._embed_fn = embed_fn
        self._episodic = EpisodicMemory(store_fn)
        self._procedural = ProceduralMemory(llm_client)

    async def record_episodic(
        self,
        task_id: str,
        description: str,
        actions: list[dict[str, Any]],
        outcome: str,
    ) -> MemoryEntry:
        return await self._episodic.record(task_id, description, actions, outcome)

    async def reflect(
        self,
        task_id: str,
        task_description: str,
        outcome: str,
    ) -> MemoryEntry:
        query_embedding = await self._embed_fn(task_description)
        similar = await self._search_fn(query_embedding, 5)

        if similar:
            episodes = [
                MemoryEntry(
                    memory_type=MemoryType.EPISODIC,
                    task_description=r.get("task_description", ""),
                    actions=[],
                    outcome=r.get("outcome", ""),
                    timestamp=r.get("timestamp", 0.0),
                )
                for r in similar
            ]
            pattern = await self._procedural.extract_pattern(episodes)
            entry = await self._procedural.create_procedural_entry(
                task_description, pattern, query_embedding
            )
            await self._store_fn(entry)
            return entry

        entry = MemoryEntry(
            memory_type=MemoryType.SEMANTIC,
            task_description=task_description,
            outcome=outcome,
            embedding=query_embedding,
            timestamp=time.time(),
        )
        await self._store_fn(entry)
        return entry
