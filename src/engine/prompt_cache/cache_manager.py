from __future__ import annotations

import json
import time
from typing import Any

from src.engine.prompt_cache.content_addressed import ContentAddressedCache
from src.llm.deepseek_client import DeepSeekClient
from src.utils.logger import get_logger

logger = get_logger(__name__)

_MAX_NOTES = 500  # bound the in-memory structured-notes store


class PromptCacheManager:
    def __init__(
        self,
        llm_client: DeepSeekClient,
        cache: ContentAddressedCache,
        max_cache_age: int = 3600,
    ) -> None:
        self._llm = llm_client
        self._cache = cache
        self._max_cache_age = max_cache_age
        self._notes: dict[str, dict[str, Any]] = {}

    async def get_or_compute(
        self,
        system_prompt: str,
        user_prompt: str,
        messages_key: str = "",
    ) -> str:
        """Cache-aside pattern: return cached result or compute + cache.

        Cache failures are logged and treated as cache-misses (fail-open)
        so a Redis outage never blocks the pipeline.
        """
        key = ContentAddressedCache.make_system_prompt_key(system_prompt, user_prompt)
        if messages_key:
            # Hash the extra key to avoid injection-style collisions
            import hashlib
            suffix = hashlib.sha256(messages_key.encode()).hexdigest()[:12]
            key = f"{key}:{suffix}"

        # Try cache read (fail-open)
        try:
            cached = await self._cache.get(key)
            if cached is not None:
                return cached
        except Exception:
            logger.exception("Cache read failed for key, proceeding without cache")

        # Compute
        messages = [{"role": "system", "content": system_prompt}]
        if user_prompt:
            messages.append({"role": "user", "content": user_prompt})

        response = await self._llm.chat(messages)

        # Try cache write (best-effort, fail-open)
        try:
            await self._cache.set(key, response, self._max_cache_age)
        except Exception:
            logger.exception("Cache write failed for key, continuing without caching")

        return response

    async def get_or_compute_with_summary(
        self,
        system_prompt: str,
        user_prompt: str,
        context_summary: str = "",
    ) -> str:
        enriched_prompt = user_prompt
        if context_summary:
            enriched_prompt = f"{user_prompt}\n\nContext summary:\n{context_summary}"
        return await self.get_or_compute(system_prompt, enriched_prompt)

    def store_structured_note(self, key: str, note: dict[str, Any]) -> None:
        """Store a structured note with bounded in-memory retention."""
        if len(self._notes) >= _MAX_NOTES:
            # Evict the oldest entry by insertion order
            oldest = next(iter(self._notes))
            del self._notes[oldest]
            logger.debug("Evicted oldest structured note (key={})", oldest)
        self._notes[key] = {"note": note, "timestamp": time.time()}

    def get_structured_note(self, key: str) -> dict[str, Any] | None:
        entry = self._notes.get(key)
        if entry is None:
            return None
        return entry["note"]

    def build_context_summary(self, relevant_notes: list[dict[str, Any]]) -> str:
        if not relevant_notes:
            return ""
        parts = []
        for note in relevant_notes:
            parts.append(json.dumps(note, ensure_ascii=False))
        return "\n---\n".join(parts)
