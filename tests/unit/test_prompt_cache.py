from __future__ import annotations

import hashlib
from unittest.mock import AsyncMock

import pytest

from src.engine.prompt_cache.cache_manager import PromptCacheManager
from src.engine.prompt_cache.content_addressed import ContentAddressedCache


class TestContentAddressedCache:
    def test_content_addressed_cache(self) -> None:
        key = ContentAddressedCache.compute_key("hello")
        expected = hashlib.sha256(b"hello").hexdigest()
        assert key == expected
        assert len(key) == 64

    @pytest.mark.asyncio
    async def test_content_addressed_cache_hit(self) -> None:
        store_fn = AsyncMock()
        get_fn = AsyncMock(return_value="cached response")
        cache = ContentAddressedCache(store_fn, get_fn)

        result = await cache.get("some content")
        assert result == "cached response"
        compute_key = hashlib.sha256(b"some content").hexdigest()
        get_fn.assert_awaited_once_with(f"cac:{compute_key}")


class TestPromptCacheManager:
    @pytest.mark.asyncio
    async def test_prompt_cache_manager(self) -> None:
        store_fn = AsyncMock()
        get_fn = AsyncMock(return_value=None)
        cache = ContentAddressedCache(store_fn, get_fn)

        llm = AsyncMock()
        llm.chat = AsyncMock(return_value="llm response")

        manager = PromptCacheManager(llm, cache)

        result = await manager.get_or_compute("system prompt", "user prompt")
        assert result == "llm response"
        llm.chat.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_prompt_cache_ttl_eviction(self) -> None:
        store_fn = AsyncMock()
        get_fn = AsyncMock(side_effect=["cached response", None])
        cache = ContentAddressedCache(store_fn, get_fn, default_ttl=0)

        first = await cache.get("content with ttl=0")
        assert first == "cached response"

        second = await cache.get("content with ttl=0")
        assert second is None
