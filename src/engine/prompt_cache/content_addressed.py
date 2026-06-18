from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from typing import Any


class ContentAddressedCache:
    def __init__(
        self,
        store_fn: Callable[[str, str, int], Any],
        get_fn: Callable[[str], str | None],
        default_ttl: int = 3600,
    ) -> None:
        self._store_fn = store_fn
        self._get_fn = get_fn
        self._default_ttl = default_ttl

    @staticmethod
    def compute_key(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    async def get(self, content: str) -> str | None:
        key = f"cac:{self.compute_key(content)}"
        return await self._get_fn(key)

    async def set(self, content: str, response: str, ttl: int | None = None) -> Any:
        key = f"cac:{self.compute_key(content)}"
        return await self._store_fn(key, response, ttl or self._default_ttl)

    @staticmethod
    def make_system_prompt_key(system_prompt: str, user_prompt: str) -> str:
        combined = json.dumps({"system": system_prompt, "user": user_prompt}, sort_keys=True)
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()
