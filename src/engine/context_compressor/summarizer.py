from __future__ import annotations

from src.core.types import ContextChunk
from src.llm.deepseek_client import DeepSeekClient
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Summarizer:
    def __init__(self, llm_client: DeepSeekClient) -> None:
        self._llm = llm_client

    async def chunk_and_summarize(
        self, content: str, chunk_size: int = 2000, task_context: str = ""
    ) -> list[ContextChunk]:
        chunks: list[ContextChunk] = []
        start = 0
        while start < len(content):
            end = min(start + chunk_size, len(content))
            segment = content[start:end]
            chunk = ContextChunk(content=segment, token_count=len(segment))
            chunk.summary = await self._summarize_segment(segment, task_context)
            chunks.append(chunk)
            start = end
        logger.info("Summarizer: created {} chunks from {} chars", len(chunks), len(content))
        return chunks

    async def _summarize_segment(self, segment: str, task_context: str) -> str:
        prompt = (
            f"Task context: {task_context}\n\n"
            f"Summarize the following code segment concisely:\n```\n{segment[:1500]}\n```\n"
            "Keep the summary under 100 words. Focus on key functions, classes, and logic."
        )
        return await self._llm.chat([
            {"role": "system", "content": "You are a code summarization expert."},
            {"role": "user", "content": prompt},
        ])
