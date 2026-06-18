from __future__ import annotations

from typing import Any, Sequence
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.types import MemoryEntry, MemoryType
from src.utils.embedding import cosine_similarity, normalize_vector


class MemoryRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def store(self, entry: MemoryEntry) -> str:
        embedding_str = _vector_literal(entry.embedding)
        await self.session.execute(
            text(
                f"""INSERT INTO memory_entries
                       (memory_id, memory_type, task_description, actions, outcome, embedding, procedural_pattern, timestamp)
                   VALUES (:memory_id, :memory_type, :task_description, :actions, :outcome,
                           {embedding_str}, :procedural_pattern, :timestamp)"""
            ),
            {
                "memory_id": entry.memory_id,
                "memory_type": entry.memory_type.value,
                "task_description": entry.task_description,
                "actions": str(entry.actions),
                "outcome": entry.outcome,
                "procedural_pattern": entry.procedural_pattern,
                "timestamp": entry.timestamp,
            },
        )
        await self.session.commit()
        return entry.memory_id

    async def search_similar(
        self, query_embedding: Sequence[float], top_k: int = 5
    ) -> list[dict[str, Any]]:
        embedding_str = _vector_literal(normalize_vector(query_embedding))
        result = await self.session.execute(
            text(
                f"""SELECT memory_id, memory_type, task_description, outcome, procedural_pattern, timestamp,
                           1 - (embedding <=> {embedding_str}) AS similarity
                      FROM memory_entries
                      ORDER BY embedding <=> {embedding_str}
                      LIMIT :top_k"""
            ),
            {"top_k": top_k},
        )
        return [dict(row._mapping) for row in result.fetchall()]

    async def get_by_type(self, memory_type: MemoryType, limit: int = 20) -> list[dict[str, Any]]:
        result = await self.session.execute(
            text(
                """SELECT * FROM memory_entries
                   WHERE memory_type = :memory_type
                   ORDER BY timestamp DESC
                   LIMIT :limit"""
            ),
            {"memory_type": memory_type.value, "limit": limit},
        )
        return [dict(row._mapping) for row in result.fetchall()]


def _vector_literal(vec: Sequence[float]) -> str:
    formatted = ",".join(str(v) for v in vec)
    return f"'[{formatted}]'::vector"
