from __future__ import annotations

from typing import Any, Callable

from src.core.types import ContextChunk
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Decompressor:
    def __init__(self, get_fn: Callable[[str], Any]) -> None:
        self._get_fn = get_fn

    async def decompress(self, chunks: list[ContextChunk]) -> list[ContextChunk]:
        restored: list[ContextChunk] = []
        for chunk in chunks:
            if chunk.compressed and chunk.storage_key:
                content = await self._get_fn(chunk.storage_key)
                if content is not None:
                    chunk.content = content
                    chunk.compressed = False
                    logger.debug("Decompressed chunk {} from {}", chunk.chunk_id, chunk.storage_key)
                else:
                    logger.warning("Chunk {} not found in storage, using summary", chunk.chunk_id)
            restored.append(chunk)
        return restored
