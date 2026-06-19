from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


def parse_tool_calls(tool_calls_data: list[dict[str, Any]]) -> list[ToolCall]:
    result: list[ToolCall] = []
    for tc in tool_calls_data:
        func = tc.get("function", {})
        args_raw = func.get("arguments", "{}")
        try:
            args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
        except json.JSONDecodeError:
            args = {}
        result.append(
            ToolCall(
                id=tc.get("id", ""),
                name=func.get("name", ""),
                arguments=args,
            )
        )
    return result


def build_tool_schema() -> list[dict[str, Any]]:
    """Build the tool definitions exposed to the LLM for ReAct-loop execution.

    Each tool includes a detailed description with usage examples so the
    model can select the right tool and format arguments correctly.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": (
                    "Read the full content of a file in the codebase. "
                    "Use this before modifying any file to understand its current state. "
                    'Example: read_file(path="src/utils.py")'
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Relative or absolute path to the file",
                        },
                    },
                    "required": ["path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": (
                    "Write content to a file in the codebase, overwriting if it exists. "
                    "Always read the file first to understand its current state. "
                    "Provide the COMPLETE file content, not just a diff. "
                    'Example: write_file(path="src/utils.py", content="def foo():\\n    pass")'
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Relative or absolute path to the file",
                        },
                        "content": {
                            "type": "string",
                            "description": "Complete new content of the file",
                        },
                    },
                    "required": ["path", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "run_command",
                "description": (
                    "Run an allowed shell command. "
                    "Allowed commands: pytest, python, pip, git, ls, cat, echo. "
                    "Use this to run tests or inspect the environment. "
                    'Example: run_command(command="pytest tests/ -x --tb=short")'
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "Command to run (must start with an allowed executable)",
                        },
                    },
                    "required": ["command"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_codebase",
                "description": (
                    "Search Python files for a regex pattern. "
                    "Use this to find function definitions, class usages, import statements, etc. "
                    'Example: search_codebase(pattern="def test_", path="tests/")'
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "Regex pattern to search for (case-insensitive)",
                        },
                        "path": {
                            "type": "string",
                            "description": "Directory to search (defaults to codebase root)",
                        },
                    },
                    "required": ["pattern"],
                },
            },
        },
    ]
