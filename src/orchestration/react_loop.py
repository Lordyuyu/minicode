from __future__ import annotations

from typing import Any
from src.core.state import AgentState
from src.core.types import AgentPhase
from src.llm.deepseek_client import DeepSeekClient
from src.llm.tool_parser import parse_tool_calls, build_tool_schema
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ReActLoop:
    def __init__(self, llm_client: DeepSeekClient, max_iterations: int = 10) -> None:
        self._llm = llm_client
        self._max_iterations = max_iterations
        self._tools = build_tool_schema()

    async def run(self, state: AgentState, system_prompt: str) -> AgentState:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Task: {state.input_codebase_path}\nCurrent phase: {state.current_phase.value}"},
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
        import subprocess
        import sys
        try:
            if name == "read_file":
                path = args.get("path", "")
                with open(path, encoding="utf-8") as f:
                    return f.read()
            elif name == "write_file":
                path = args.get("path", "")
                content = args.get("content", "")
                with open(path, "w", encoding="utf-8") as f:
                    f.write(content)
                return f"Written {len(content)} bytes to {path}"
            elif name == "run_command":
                command = args.get("command", "")
                result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
                return f"Exit code: {result.returncode}\nstdout: {result.stdout[:1000]}\nstderr: {result.stderr[:500]}"
            elif name == "search_codebase":
                import re
                pattern = args.get("pattern", "")
                path = args.get("path", state.input_codebase_path)
                matches = []
                import os
                for root, _, files in os.walk(path):
                    for fname in files:
                        if fname.endswith(".py"):
                            fpath = os.path.join(root, fname)
                            with open(fpath, encoding="utf-8") as f:
                                for i, line in enumerate(f, 1):
                                    if re.search(pattern, line):
                                        matches.append(f"{fpath}:{i}: {line.rstrip()}")
                return "\n".join(matches)
        except Exception as e:
            return f"Tool {name} failed: {e}"
        return f"Unknown tool: {name}"
