from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.types import Skill
from src.storage.vector_utils import vector_literal
from src.utils.embedding import normalize_vector


class SkillRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def register_skill(self, skill: Skill) -> str:
        skill_id = skill.skill_id or str(uuid.uuid4())
        embedding_str = vector_literal(skill.embedding)
        await self.session.execute(
            text(
                f"""INSERT INTO skills
                       (skill_id, name, category, description, embedding,
                        success_rate, invocation_count)
                   VALUES (:skill_id, :name, :category, :description,
                           {embedding_str}, :success_rate, :invocation_count)
                   ON CONFLICT (name) DO UPDATE SET
                       embedding = {embedding_str}, updated_at = :now"""
            ),
            {
                "skill_id": skill_id,
                "name": skill.name,
                "category": skill.category.value,
                "description": skill.description,
                "success_rate": skill.success_rate,
                "invocation_count": skill.invocation_count,
                "now": datetime.now(UTC),
            },
        )
        await self.session.commit()
        return skill_id

    async def search_similar(
        self, query_embedding: Sequence[float], top_k: int = 10
    ) -> list[dict[str, Any]]:
        embedding_str = vector_literal(normalize_vector(query_embedding))
        result = await self.session.execute(
            text(
                f"""SELECT skill_id, name, category, description, success_rate, invocation_count,
                           1 - (embedding <=> {embedding_str}) AS similarity
                      FROM skills
                      ORDER BY embedding <=> {embedding_str}
                      LIMIT :top_k"""
            ),
            {"top_k": top_k},
        )
        rows = await result.fetchall()
        return [dict(row._mapping) for row in rows]

    async def record_invocation(
        self, task_id: str, skill_id: str, success: bool, duration_ms: int,
    ) -> None:
        await self.session.execute(
            text(
                """INSERT INTO skill_invocations (task_id, skill_id, success, duration_ms)
                   VALUES (:task_id, :skill_id, :success, :duration_ms)"""
            ),
            {"task_id": task_id, "skill_id": skill_id, "success": success, "duration_ms": duration_ms},
        )
        await self.session.commit()
