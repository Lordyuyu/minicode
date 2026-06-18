from __future__ import annotations

import time
from typing import Any, Sequence
from openai import AsyncOpenAI
from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DeepSeekClient:
    def __init__(self, api_key: str | None = None) -> None:
        self._client = AsyncOpenAI(
            api_key=api_key or settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
        )
        self._model = settings.deepseek_model

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        start = time.monotonic()
        kwargs = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools

        response = await self._client.chat.completions.create(**kwargs)
        elapsed = time.monotonic() - start
        logger.debug("DeepSeek chat call completed in {:.2f}s", elapsed)

        return response.choices[0].message.content or ""

    async def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> tuple[str, list[dict[str, Any]]]:
        start = time.monotonic()
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        elapsed = time.monotonic() - start
        logger.debug("DeepSeek tool call completed in {:.2f}s", elapsed)

        message = response.choices[0].message
        content = message.content or ""
        tool_calls = []
        if message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append({
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                })
        return content, tool_calls

    async def embed(self, text: str) -> list[float]:
        start = time.monotonic()
        response = await self._client.embeddings.create(
            model="text-embedding-ada-002",
            input=text,
        )
        elapsed = time.monotonic() - start
        logger.debug("DeepSeek embedding call completed in {:.2f}s", elapsed)
        return response.data[0].embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        start = time.monotonic()
        response = await self._client.embeddings.create(
            model="text-embedding-ada-002",
            input=texts,
        )
        elapsed = time.monotonic() - start
        logger.debug("DeepSeek batch embedding completed in {:.2f}s", elapsed)
        return [d.embedding for d in response.data]
