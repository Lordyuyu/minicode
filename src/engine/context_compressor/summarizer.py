from __future__ import annotations

from src.core.types import ContextChunk
from src.llm.deepseek_client import DeepSeekClient
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Approximate tokens-to-chars ratio for code (rough heuristic: ~4 chars/token)
_CHARS_PER_TOKEN = 4


class Summarizer:
    def __init__(self, llm_client: DeepSeekClient) -> None:
        self._llm = llm_client

    async def chunk_and_summarize(
        self, content: str, chunk_size: int = 2000, task_context: str = ""
    ) -> list[ContextChunk]:
        """Split *content* into fixed-size chunks and summarise each one.

        The full *chunk_size* is used for both storage and summarisation
        (previously truncated at 1500 chars regardless of *chunk_size*).
        """
        chunks: list[ContextChunk] = []
        start = 0
        while start < len(content):
            end = min(start + chunk_size, len(content))
            segment = content[start:end]
            # Approximate token count (characters / 4 for code)
            chunk = ContextChunk(
                content=segment,
                token_count=max(1, len(segment) // _CHARS_PER_TOKEN),
            )
            try:
                chunk.summary = await self._summarize_segment(segment, task_context, chunk_size)
            except Exception:
                logger.exception("Summarisation failed for segment, using truncated content as summary")
                chunk.summary = segment[:200] + "…" if len(segment) > 200 else segment
            chunks.append(chunk)
            start = end
        logger.info("Summarizer: created {} chunks from {} chars", len(chunks), len(content))
        return chunks

    async def _summarize_segment(
        self, segment: str, task_context: str, chunk_size: int,
    ) -> str:
        """Summarise a single content segment using the LLM."""
        # Pass the full segment (previously truncated at hard-coded 1500)
        prompt = (
            f"Task context: {task_context}\n\n"
            f"Summarize the following code segment concisely:\n"
            f"```\n{segment[:chunk_size]}\n```\n"
            "Keep the summary under 100 words. Focus on key functions, classes, and logic."
        )
        return await self._llm.chat([
            {"role": "system", "content": "You are a code summarization expert."},
            {"role": "user", "content": prompt},
        ])
