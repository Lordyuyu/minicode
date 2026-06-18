from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


class TaskRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create_task(
        self, codebase_path: str, test_command: str = "pytest"
    ) -> dict[str, Any]:
        task_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        await self.session.execute(
            text(
                """INSERT INTO tasks (task_id, input_codebase_path, test_command, status, created_at, updated_at)
                   VALUES (:task_id, :codebase_path, :test_command, 'pending', :now, :now)"""
            ),
            {"task_id": task_id, "codebase_path": codebase_path, "test_command": test_command, "now": now},
        )
        await self.session.commit()
        return {"task_id": task_id, "status": "pending"}

    async def get_task(self, task_id: str) -> dict[str, Any] | None:
        result = await self.session.execute(
            text("SELECT * FROM tasks WHERE task_id = :task_id"),
            {"task_id": task_id},
        )
        row = result.fetchone()
        if row is None:
            return None
        return dict(row._mapping)

    async def update_status(self, task_id: str, status: str) -> None:
        now = datetime.now(timezone.utc)
        await self.session.execute(
            text("UPDATE tasks SET status = :status, updated_at = :now WHERE task_id = :task_id"),
            {"status": status, "task_id": task_id, "now": now},
        )
        await self.session.commit()
