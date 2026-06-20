from __future__ import annotations

import os
import re
import shlex
import subprocess
from typing import Any

from src.core.state import AgentState
from src.llm.deepseek_client import DeepSeekClient
from src.llm.tool_parser import build_tool_schema, parse_tool_calls
from src.utils.logger import get_logger

logger = get_logger(__name__)

# --- Command allowlist for run_command tool ---
# Only these executables may be invoked by an LLM-generated command.
# Each entry is a (name, allow_args) pair; allow_args=True means
# arguments are unrestricted, False means no arguments allowed.
_ALLOWED_COMMANDS: dict[str, bool] = {
    "pytest": True,
    "python": True,
    "pip": True,
    "git": True,
    "ls": False,
    "dir": False,
    "cat": False,
    "type": False,
    "echo": False,
}

_MAX_STDOUT = 2000
_MAX_STDERR = 1000
_CMD_TIMEOUT = 30


def _sanitize_path(requested: str, sandbox_root: str) -> str:
    """Resolve *requested* path and enforce it stays under *sandbox_root*.

    Returns the resolved absolute path if safe, otherwise raises ``ValueError``.
    """
    if not requested:
        raise ValueError("Empty path not allowed")
    # Resolve to absolute, normalising '..' and symlinks
    resolved = os.path.normpath(os.path.join(sandbox_root, requested))
    sandbox = os.path.normpath(os.path.abspath(sandbox_root))
    if not resolved.startswith(sandbox + os.sep) and resolved != sandbox:
        raise ValueError(
            f"Path traversal denied: {requested!r} resolves outside sandbox"
        )
    return resolved


class ReActLoop:
    def __init__(self, llm_client: DeepSeekClient, max_iterations: int = 10) -> None:
        self._llm = llm_client
        self._max_iterations = max_iterations
        self._tools = build_tool_schema()

    async def run(self, state: AgentState, system_prompt: str) -> AgentState:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Task: {state.input_codebase_path}\n"
                f"Current phase: {state.current_phase.value}",
            },
        ]

        for iteration in range(self._max_iterations):
            content, tool_calls = await self._llm.chat_with_tools(
                messages, tools=self._tools
            )
            messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})

            if not tool_calls:
                logger.info("ReAct loop finished after {} iterations", iteration + 1)
                state.metadata["react_final_output"] = content
                break

            for tc in parse_tool_calls(tool_calls):
                result = await self._execute_tool(tc.name, tc.arguments, state)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": str(result),
                })
        else:
            logger.warning("ReAct loop reached max iterations ({})", self._max_iterations)

        return state

    async def _execute_tool(self, name: str, args: dict[str, Any], state: AgentState) -> str:
        codebase = state.input_codebase_path
        try:
            if name == "read_file":
                path = _sanitize_path(args.get("path", ""), codebase)
                with open(path, encoding="utf-8") as f:
                    return f.read()

            elif name == "write_file":
                path = _sanitize_path(args.get("path", ""), codebase)
                content = args.get("content", "")
                os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)
                return f"Written {len(content)} bytes to {path}"

            elif name == "run_command":
                return await self._run_sandboxed_command(args.get("command", ""))

            elif name == "search_codebase":
                return self._search_codebase(
                    args.get("pattern", ""),
                    args.get("path", codebase),
                )

        except Exception as e:
            logger.warning("Tool {} failed: {}", name, e)
            return f"Tool {name} failed: {e}"

        return f"Unknown tool: {name}"

    async def _run_sandboxed_command(self, command: str) -> str:
        """Execute a shell command with allowlist enforcement and no shell
        interpolation.

        Uses ``shlex.split()`` to tokenize the command safely, then verifies
        the executable is on the allowlist before running via ``shell=False``.
        """
        if not command.strip():
            return "Error: empty command"

        try:
            tokens = shlex.split(command)
        except ValueError as e:
            return f"Error parsing command: {e}"

        if not tokens:
            return "Error: no tokens after parsing"

        exe = os.path.basename(tokens[0])
        if exe not in _ALLOWED_COMMANDS:
            return (
                f"Error: '{exe}' is not on the allowed command list. "
                f"Allowed: {', '.join(sorted(_ALLOWED_COMMANDS))}"
            )

        try:
            result = subprocess.run(
                tokens,
                shell=False,
                capture_output=True,
                text=True,
                timeout=_CMD_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            return f"Error: command timed out after {_CMD_TIMEOUT}s"
        except FileNotFoundError:
            return f"Error: executable '{tokens[0]}' not found"

        return (
            f"Exit code: {result.returncode}\n"
            f"stdout: {result.stdout[:_MAX_STDOUT]}\n"
            f"stderr: {result.stderr[:_MAX_STDERR]}"
        )

    @staticmethod
    def _search_codebase(pattern: str, search_path: str) -> str:
        """Search Python files for a regex *pattern* under *search_path*.

        The pattern is compiled with ``re.IGNORECASE``.  If compilation fails
        the error is returned as a string so the agent can retry with a
        corrected pattern.
        """
        if not pattern:
            return "Error: empty search pattern"

        try:
            compiled = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            return f"Error: invalid regex pattern: {e}"

        matches: list[str] = []
        for root, _, files in os.walk(search_path):
            for fname in files:
                if fname.endswith(".py"):
                    fpath = os.path.join(root, fname)
                    try:
                        with open(fpath, encoding="utf-8") as f:
                            for i, line in enumerate(f, 1):
                                if compiled.search(line):
                                    matches.append(f"{fpath}:{i}: {line.rstrip()}")
                    except OSError:
                        continue

        if not matches:
            return "No matches found"
        return "\n".join(matches[:200])  # cap at 200 results
