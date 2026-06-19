"""
Pre-defined bug scenarios for benchmarking.

Each scenario is a self-contained Python module (as a string) plus a test
that exercises the bug.  The ``BugScenario`` dataclass stores everything
needed to set up a temp directory, run the pipeline, and verify the fix.

Expected-string checks are intentionally lenient — they only verify that
the fix *direction* is correct (e.g. a bounds check was added) rather
than enforcing a specific implementation style.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BugScenario:
    """A single known-bug test case for the evaluation framework."""

    name: str
    description: str
    source_files: dict[str, str]
    test_file: str
    expected_patch_contains: list[str] = field(default_factory=list)
    expected_patch_not_contains: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

SCENARIOS: list[BugScenario] = [
    BugScenario(
        name="divide-by-zero",
        description="Function divide(a, b) does not handle b=0.",
        source_files={
            "calculator/__init__.py": "",
            "calculator/ops.py": (
                "def divide(a, b):\n    return a / b\n"
            ),
        },
        test_file=(
            "import pytest\n"
            "from calculator.ops import divide\n\n"
            "def test_divide_by_zero():\n"
            "    # Buggy code raises ZeroDivisionError; fix should raise ValueError\n"
            "    with pytest.raises(ValueError):\n"
            "        divide(10, 0)\n\n"
            "def test_divide_normal():\n"
            "    assert divide(10, 2) == 5.0\n"
        ),
        expected_patch_contains=["if"],
        expected_patch_not_contains=[],
        tags=["arithmetic", "edge-case"],
    ),
    BugScenario(
        name="index-out-of-range",
        description="get_nth(items, n) accesses items[n] without bounds check.",
        source_files={
            "utils/__init__.py": "",
            "utils/accessors.py": (
                "def get_nth(items, n):\n    return items[n]\n"
            ),
        },
        test_file=(
            "from utils.accessors import get_nth\n\n"
            "def test_index_error():\n"
            "    result = get_nth([1, 2, 3], 5)\n"
            "    assert result is None\n"
        ),
        expected_patch_contains=["len"],  # any bounds check uses len()
        expected_patch_not_contains=[],
        tags=["bounds", "edge-case"],
    ),
    BugScenario(
        name="none-attribute-access",
        description="format_name(user) calls user.name without checking for None.",
        source_files={
            "models/__init__.py": "",
            "models/user.py": (
                "class User:\n"
                "    def __init__(self, name: str | None):\n"
                "        self.name = name\n\n\n"
                "def format_name(user: User | None) -> str:\n"
                "    return f'User: {user.name}'\n"
            ),
        },
        test_file=(
            "from models.user import format_name\n\n"
            "def test_none_user():\n"
            "    result = format_name(None)\n"
            "    assert result == 'User: Unknown'\n"
        ),
        expected_patch_contains=["None"],  # any None-check pattern
        expected_patch_not_contains=[],
        tags=["null-safety", "type-error"],
    ),
    BugScenario(
        name="key-error-missing",
        description="get_config(key) uses dict[key] which raises KeyError for missing keys.",
        source_files={
            "config/__init__.py": "",
            "config/reader.py": (
                "DEFAULTS = {'host': 'localhost', 'port': 8080}\n\n\n"
                "def get_config(key: str):\n"
                "    return DEFAULTS[key]\n"
            ),
        },
        test_file=(
            "from config.reader import get_config\n\n"
            "def test_missing_key():\n"
            "    result = get_config('timeout')\n"
            "    assert result is None\n\n"
            "def test_existing_key():\n"
            "    assert get_config('host') == 'localhost'\n"
        ),
        expected_patch_contains=[".get"],  # use .get() for safe access
        expected_patch_not_contains=[],
        tags=["dict-access", "edge-case"],
    ),
    BugScenario(
        name="type-error-concatenation",
        description="combine(a, b) uses + which fails when one operand is not a string.",
        source_files={
            "formatter/__init__.py": "",
            "formatter/core.py": (
                "def combine(a, b):\n    return a + b\n"
            ),
        },
        test_file=(
            "from formatter.core import combine\n\n"
            "def test_combine_mixed_types():\n"
            "    result = combine('count: ', 42)\n"
            "    assert result == 'count: 42'\n"
        ),
        expected_patch_contains=["str"],  # must convert via str()
        expected_patch_not_contains=[],
        tags=["type-cast", "type-error"],
    ),
    BugScenario(
        name="mutable-default-arg",
        description="append_item(item, target=[]) uses mutable default argument.",
        source_files={
            "helpers.py": (
                "def append_item(item, target=[]):\n"
                "    target.append(item)\n"
                "    return target\n"
            ),
        },
        test_file=(
            "from helpers import append_item\n\n"
            "def test_default_arg_isolation():\n"
            "    a = append_item(1)\n"
            "    b = append_item(2)\n"
            "    assert a == [1]\n"
            "    assert b == [2]\n"
        ),
        expected_patch_contains=["None"],  # must use None sentinel
        expected_patch_not_contains=["target=[]"],  # must remove mutable default
        tags=["python-gotcha", "mutable-default"],
    ),
    BugScenario(
        name="infinite-recursion",
        description="factorial(n) missing base case for n <= 1.",
        source_files={
            "mathlib/__init__.py": "",
            "mathlib/factorial.py": (
                "def factorial(n: int) -> int:\n"
                "    return n * factorial(n - 1)\n"
            ),
        },
        test_file=(
            "from mathlib.factorial import factorial\n\n"
            "def test_factorial_zero():\n"
            "    assert factorial(0) == 1\n\n"
            "def test_factorial_five():\n"
            "    assert factorial(5) == 120\n"
        ),
        expected_patch_contains=["if"],  # must have a base-case conditional
        expected_patch_not_contains=[],
        tags=["recursion", "edge-case"],
    ),
]
