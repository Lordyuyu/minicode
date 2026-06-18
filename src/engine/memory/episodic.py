from __future__ import annotations

import time
from typing import Any, Callable

from src.core.types import MemoryEntry, MemoryType


class EpisodicMemory:
    def __init__(self, store_fn: Callable[[MemoryEntry], Any]) -> None:
        self._store_fn = store_fn

    async def record(
        self,
        task_id: str,
        description: str,
        actions: list[dict[str, Any]],
        outcome: str,
    ) -> MemoryEntry:
        entry = MemoryEntry(
            memory_type=MemoryType.EPISODIC,
            task_description=description,
            actions=actions,
            outcome=outcome,
            timestamp=time.time(),
        )
        await self._store_fn(entry)
        return entry
