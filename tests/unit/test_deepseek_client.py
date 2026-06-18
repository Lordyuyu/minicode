from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch
from src.llm.deepseek_client import DeepSeekClient
from src.llm.tool_parser import ToolCall, parse_tool_calls


@pytest.mark.asyncio
async def test_deepseek_chat():
    client = DeepSeekClient(api_key="test-key")
    client._client.chat.completions.create = AsyncMock()
    client._client.chat.completions.create.return_value = AsyncMock()
    client._client.chat.completions.create.return_value.choices = [
        AsyncMock(message=AsyncMock(content="Hello!", tool_calls=[]))
    ]
    response = await client.chat([{"role": "user", "content": "Hi"}])
    assert response == "Hello!"


@pytest.mark.asyncio
async def test_deepseek_embed():
    client = DeepSeekClient(api_key="test-key")
    client._client.embeddings.create = AsyncMock()
    client._client.embeddings.create.return_value = AsyncMock()
    client._client.embeddings.create.return_value.data = [AsyncMock(embedding=[0.1, 0.2, 0.3])]
    embedding = await client.embed("test text")
    assert len(embedding) == 3


def test_parse_tool_calls():
    calls = parse_tool_calls(
        [
            {
                "id": "call_1",
                "function": {"name": "read_file", "arguments": '{"path": "test.py"}'},
            }
        ]
    )
    assert len(calls) == 1
    assert calls[0].name == "read_file"
    assert calls[0].arguments["path"] == "test.py"


def test_parse_tool_calls_empty():
    assert parse_tool_calls([]) == []
