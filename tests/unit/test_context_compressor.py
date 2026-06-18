from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from src.core.types import ContextChunk
from src.engine.context_compressor.decompressor import Decompressor
from src.engine.context_compressor.hierarchical import HierarchicalCompressor
from src.engine.context_compressor.summarizer import Summarizer


@pytest.mark.asyncio
async def test_summarizer_chunk_and_summarize():
    llm = AsyncMock()
    llm.chat = AsyncMock(return_value="summary text")
    summarizer = Summarizer(llm)
    chunks = await summarizer.chunk_and_summarize("long code content " * 1000, chunk_size=500)
    assert len(chunks) > 0
    assert all(c.summary for c in chunks)


@pytest.mark.asyncio
async def test_compressor_compress():
    store_fn = AsyncMock()
    compressor = HierarchicalCompressor(AsyncMock(), store_fn)
    compressor._summarizer.chunk_and_summarize = AsyncMock(
        return_value=[ContextChunk(content="test", summary="summary")]
    )
    result = await compressor.compress("some long context", "task-1")
    assert "summary" in result or len(result) < len("some long context")


@pytest.mark.asyncio
async def test_decompressor_decompress():
    get_fn = AsyncMock(return_value="full content")
    decompressor = Decompressor(get_fn)
    chunks = [ContextChunk(chunk_id="c1", compressed=True, storage_key="chunk:c1", summary="summary text")]
    result = await decompressor.decompress(chunks)
    assert len(result) == 1
    assert result[0].content == "full content"
    assert result[0].compressed is False
