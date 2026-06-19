from __future__ import annotations

import asyncio
import time
from typing import Any

from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError,
)

from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Retry configuration
_RETRY_BASE_DELAY = 1.0  # seconds
_RETRY_MAX_DELAY = 30.0  # seconds
_RETRY_JITTER = 0.1  # ±10% jitter

# HTTP status codes that warrant a retry
_RETRYABLE_STATUSES: set[int] = {429, 500, 502, 503, 504}


def _backoff_delay(attempt: int) -> float:
    """Exponential backoff with jitter.

    Formula: min(base * 2^attempt, max) * (1 ± jitter)
    """
    import random

    raw = min(_RETRY_BASE_DELAY * (2 ** attempt), _RETRY_MAX_DELAY)
    jitter = 1.0 + random.uniform(-_RETRY_JITTER, _RETRY_JITTER)
    return raw * jitter


def _is_retryable(exc: Exception) -> bool:
    """Determine whether an exception warrants a retry."""
    if isinstance(exc, RateLimitError):
        return True
    if isinstance(exc, APITimeoutError):
        return True
    if isinstance(exc, APIConnectionError):
        return True
    if isinstance(exc, APIError):
        status = getattr(exc, "status_code", None)
        return status in _RETRYABLE_STATUSES
    return False


class DeepSeekClient:
    def __init__(self, api_key: str | None = None) -> None:
        self._client = AsyncOpenAI(
            api_key=api_key or settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            timeout=float(settings.llm_timeout),
            max_retries=0,  # we handle retries ourselves
        )
        self._model = settings.deepseek_model
        self._embedding_model = settings.embedding_model
        self._max_retries = settings.llm_max_retries

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        """Send a chat completion request with automatic retry on transient errors."""
        response = await self._retryable_call(
            lambda: self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
            ),
            label="chat",
        )
        return response.choices[0].message.content or ""

    async def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Send a chat completion request expecting tool calls in the response."""
        response = await self._retryable_call(
            lambda: self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
            ),
            label="chat_with_tools",
        )

        message = response.choices[0].message
        content = message.content or ""
        tool_calls: list[dict[str, Any]] = []
        if message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append(
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                )
        return content, tool_calls

    async def embed(self, text: str) -> list[float]:
        """Compute an embedding vector for a single text."""
        response = await self._retryable_call(
            lambda: self._client.embeddings.create(
                model=self._embedding_model,
                input=text,
            ),
            label="embed",
        )
        return response.data[0].embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Compute embedding vectors for a batch of texts."""
        response = await self._retryable_call(
            lambda: self._client.embeddings.create(
                model=self._embedding_model,
                input=texts,
            ),
            label="embed_batch",
        )
        return [d.embedding for d in response.data]

    # ------------------------------------------------------------------
    # Retry machinery
    # ------------------------------------------------------------------

    async def _retryable_call(self, fn, *, label: str) -> Any:
        """Call *fn* with exponential-backoff retry on transient failures.

        Args:
            fn: Async callable that performs the API request.
            label: Human-readable label for log messages.

        Returns:
            The API response object on success.

        Raises:
            The last captured exception after all retries are exhausted.
        """
        last_exc: Exception | None = None
        start = time.monotonic()

        for attempt in range(self._max_retries + 1):
            try:
                response = await fn()
                elapsed = time.monotonic() - start

                # Log token usage when available
                usage = getattr(response, "usage", None)
                if usage:
                    logger.debug(
                        "LLM {} call: {:.2f}s, attempt={}, "
                        "prompt_tokens={}, completion_tokens={}, total_tokens={}",
                        label,
                        elapsed,
                        attempt + 1,
                        getattr(usage, "prompt_tokens", "?"),
                        getattr(usage, "completion_tokens", "?"),
                        getattr(usage, "total_tokens", "?"),
                    )
                else:
                    logger.debug(
                        "LLM {} call completed in {:.2f}s (attempt {})",
                        label,
                        elapsed,
                        attempt + 1,
                    )
                return response

            except Exception as exc:
                last_exc = exc
                if attempt >= self._max_retries or not _is_retryable(exc):
                    logger.error(
                        "LLM {} call failed (attempt {}): {}",
                        label,
                        attempt + 1,
                        exc,
                    )
                    raise

                delay = _backoff_delay(attempt)
                logger.warning(
                    "LLM {} call failed (attempt {}), retrying in {:.1f}s: {}",
                    label,
                    attempt + 1,
                    delay,
                    exc,
                )
                await asyncio.sleep(delay)

        # Should be unreachable — the loop above either returns or raises
        raise last_exc  # type: ignore[misc]
