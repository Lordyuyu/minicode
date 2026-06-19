from __future__ import annotations

import json
from typing import Any

from redis.asyncio import Redis

from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RedisClient:
    def __init__(self) -> None:
        self._client: Redis | None = None

    async def initialize(self) -> None:
        self._client = Redis.from_url(
            settings.redis_url,
            decode_responses=True,
            max_connections=20,
        )
        await self._client.ping()
        logger.info("Redis connection established")

    async def set_cache(self, key: str, value: Any, ttl: int = 300) -> None:
        if self._client is None:
            raise RuntimeError("Redis not initialized")
        serialized = json.dumps(value) if not isinstance(value, str) else value
        await self._client.setex(key, ttl, serialized)

    async def get_cache(self, key: str) -> Any | None:
        if self._client is None:
            raise RuntimeError("Redis not initialized")
        value = await self._client.get(key)
        if value is None:
            return None
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value

    async def delete_cache(self, key: str) -> None:
        if self._client is None:
            raise RuntimeError("Redis not initialized")
        await self._client.delete(key)

    async def store_compressed_chunk(
        self, chunk_id: str, content: str, ttl: int = 3600
    ) -> None:
        await self.set_cache(f"chunk:{chunk_id}", content, ttl)

    async def get_compressed_chunk(self, chunk_id: str) -> str | None:
        return await self.get_cache(f"chunk:{chunk_id}")

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            logger.info("Redis connection closed")
