from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy import text

from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

_engine = create_async_engine(
    settings.postgres_dsn,
    echo=False,
    pool_size=5,
    max_overflow=10,
)

_session_factory = async_sessionmaker(_engine, expire_on_commit=False)


class PostgresClient:
    def __init__(self) -> None:
        self._engine = _engine
        self._session_factory = _session_factory

    async def initialize(self) -> None:
        async with self._engine.begin() as conn:
            await conn.execute(text("SELECT 1"))
        logger.info("PostgreSQL connection established")

    async def run_migrations(self) -> None:
        from pathlib import Path
        sql_path = Path(__file__).parent / "migrations" / "001_init.sql"
        sql = sql_path.read_text(encoding="utf-8")
        async with self._engine.begin() as conn:
            for statement in sql.split(";"):
                stripped = statement.strip()
                if stripped:
                    await conn.execute(text(stripped))
        logger.info("Database migrations applied")

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        async with self._session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()

    async def close(self) -> None:
        await self._engine.dispose()
        logger.info("PostgreSQL connection closed")
