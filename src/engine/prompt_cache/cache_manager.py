from __future__ import annotations

import json
import time
from typing import Any

from src.engine.prompt_cache.content_addressed import ContentAddressedCache
from src.llm.deepseek_client import DeepSeekClient


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
        key = ContentAddressedCache.make_system_prompt_key(system_prompt, user_prompt)
        if messages_key:
            key = f"{key}:{messages_key}"

        cached = await self._cache.get(key)
        if cached is not None:
            return cached

        messages = [{"role": "system", "content": system_prompt}]
        if user_prompt:
            messages.append({"role": "user", "content": user_prompt})

        response = await self._llm.chat(messages)
        await self._cache.set(key, response, self._max_cache_age)
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
