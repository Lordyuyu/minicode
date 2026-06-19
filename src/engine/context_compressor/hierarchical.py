from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.engine.context_compressor.summarizer import Summarizer
from src.utils.logger import get_logger

logger = get_logger(__name__)

_MAX_RECURSION_DEPTH = 3  # prevent unbounded stack growth


class HierarchicalCompressor:
    def __init__(
        self,
        llm_client: Any,
        store_fn: Callable[[str, str], Any],
        chunk_size: int = 2000,
    ) -> None:
        self._summarizer = Summarizer(llm_client)
        self._store_fn = store_fn
        self._chunk_size = chunk_size

    async def compress(
        self, content: str, task_id: str, task_context: str = "",
    ) -> str:
        """Public entry point — delegates to the recursive implementation
        with a depth guard."""
        return await self._compress(content, task_id, task_context, depth=0)

    async def _compress(
        self, content: str, task_id: str, task_context: str, *, depth: int,
    ) -> str:
        if depth >= _MAX_RECURSION_DEPTH:
            logger.warning(
                "Hierarchical compressor hit max recursion depth ({}), "
                "returning truncated content",
                _MAX_RECURSION_DEPTH,
            )
            return content[:_MAX_RECURSION_DEPTH * self._chunk_size]

        chunks = await self._summarizer.chunk_and_summarize(
            content, self._chunk_size, task_context
        )
        summaries: list[str] = []
        for i, chunk in enumerate(chunks):
            storage_key = f"chunk:{task_id}:{i}"
            try:
                await self._store_fn(storage_key, chunk.content)
            except Exception:
                logger.exception("Failed to store chunk {}", storage_key)
            chunk.storage_key = storage_key
            chunk.compressed = True
            chunk.content = ""
            summaries.append(chunk.summary)

        compressed = "\n".join(
            f"[Chunk {i}] {s}" for i, s in enumerate(summaries)
        )

        if len(compressed) > self._chunk_size * 2 and depth + 1 < _MAX_RECURSION_DEPTH:
            logger.info("Recursive compression (depth {}) — {} chars", depth + 1, len(compressed))
            compressed = await self._compress(
                compressed, f"{task_id}_meta", "compressed summaries", depth=depth + 1,
            )

        logger.info("Compressed {} chars -> {} chars (depth {})", len(content), len(compressed), depth)
        return compressed
