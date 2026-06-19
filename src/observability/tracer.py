"""
LangSmith integration for MiniCode observability.

Provides trace decorators and context managers for key MiniCode
functions: LLM calls, graph node execution, agent runs, and the
top-level pipeline.  When ``LANGSMITH_API_KEY`` is configured the
traces are sent to LangSmith automatically; when left empty all
decorators are no-ops so the code requires no conditional branching.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from src.config.settings import settings


def _is_enabled() -> bool:
    return bool(settings.langsmith_api_key)


def setup_langsmith() -> None:
    """Initialize LangSmith tracing.

    Safe to call multiple times — only configures environment on the
    first call.  Must be called before any traced function executes.
    """
    if not _is_enabled():
        return

    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_API_KEY", settings.langsmith_api_key)
    os.environ.setdefault("LANGCHAIN_PROJECT", settings.langsmith_project)
    os.environ.setdefault("LANGCHAIN_ENDPOINT", settings.langsmith_endpoint)

    # Try importing langsmith and verify connectivity (best-effort)
    try:
        import langsmith  # noqa: F401
    except ImportError:
        pass  # tracing will be skipped if the package isn't installed


def traced(name: str | None = None, *, tags: list[str] | None = None):
    """Decorator that wraps a function with a LangSmith trace span.

    When LangSmith is not configured (no API key set), the original
    function is returned unchanged.

    Usage::

        @traced("bug_locator.locate")
        async def locate(self, state: AgentState) -> list[BugReport]:
            ...

    Args:
        name: Human-readable run name shown in the LangSmith UI.
            Defaults to ``module.qualname`` of the wrapped function.
        tags: Optional list of string tags to attach to every run.
    """

    def decorator(fn: Callable) -> Callable:
        if not _is_enabled():
            return fn

        try:
            from langsmith import traceable  # type: ignore[import-untyped]

            trace_kwargs: dict[str, Any] = {}
            if name:
                trace_kwargs["name"] = name
            if tags:
                trace_kwargs["tags"] = tags

            return traceable(**trace_kwargs)(fn)
        except ImportError:
            return fn

    return decorator
