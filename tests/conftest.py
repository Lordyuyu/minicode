from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any
import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.fixture
def mock_deepseek_client():
    client = MagicMock()
    client.chat = AsyncMock(return_value={"choices": [{"message": {"content": "mock response", "tool_calls": []}}]})
    client.embed = AsyncMock(return_value=[[0.1] * 768])
    return client


@pytest.fixture
def mock_postgres():
    from unittest.mock import AsyncMock
    session = AsyncMock()
    session.execute = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    return session


@pytest.fixture
def sample_codebase(tmp_path):
    pkg = tmp_path / "sample_project"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "calculator.py").write_text(
        "def add(a, b):\n    return a + b\n\ndef divide(a, b):\n    return a / b\n"
    )
    (pkg / "test_calculator.py").write_text(
        "from sample_project.calculator import divide\n\ndef test_divide_by_zero():\n    divide(1, 0)\n"
    )
    return str(tmp_path)
