from __future__ import annotations

from typing import Any, Callable

from src.core.types import ContextChunk
from src.engine.context_compressor.summarizer import Summarizer
from src.utils.logger import get_logger

logger = get_logger(__name__)


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
        self, content: str, task_id: str, task_context: str = ""
    ) -> str:
        chunks = await self._summarizer.chunk_and_summarize(
            content, self._chunk_size, task_context
        )
        summaries: list[str] = []
        for i, chunk in enumerate(chunks):
            storage_key = f"chunk:{task_id}:{i}"
            await self._store_fn(storage_key, chunk.content)
            chunk.storage_key = storage_key
            chunk.compressed = True
            chunk.content = ""
            summaries.append(chunk.summary)

        compressed = "\n".join(
            f"[Chunk {i}] {s}" for i, s in enumerate(summaries)
        )
        if len(compressed) > self._chunk_size * 2:
            logger.info("Recursive compression needed ({} chars)", len(compressed))
            compressed = await self.compress(compressed, f"{task_id}_meta", "compressed summaries")
        logger.info("Compressed {} chars -> {} chars", len(content), len(compressed))
        return compressed
