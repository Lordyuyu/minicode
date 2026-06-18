from __future__ import annotations

import difflib
import json
import os
import re


def _extract_json(text: str) -> str:
    match = re.search(r'`(?:json)?\n(.*?)\n`', text, re.DOTALL)
    if match:
        return match.group(1)
    return text.strip()
from typing import Any

from src.core.state import AgentState
from src.core.types import BugReport, PatchResult
from src.llm.deepseek_client import DeepSeekClient
from src.llm.prompt_templates import build_patch_generation_prompt


class PatchGenerator:
    def __init__(self, llm_client: DeepSeekClient) -> None:
        self._llm = llm_client

    async def generate(self, state: AgentState) -> list[PatchResult]:
        patches: list[PatchResult] = []
        for bug in state.bug_reports:
            result = await self._generate_for_bug(bug, state.input_codebase_path)
            if result is not None:
                patches.append(result)
        state.patches = patches
        return patches

    async def _generate_for_bug(
        self, bug: BugReport, codebase_path: str
    ) -> PatchResult | None:
        file_path = os.path.join(codebase_path, bug.file_path)
        if not os.path.isfile(file_path):
            file_path = bug.file_path

        try:
            with open(file_path, encoding="utf-8") as f:
                original_content = f.read()
        except OSError:
            return None

        prompt = build_patch_generation_prompt(json.dumps({
            "file_path": bug.file_path,
            "line_start": bug.line_start,
            "line_end": bug.line_end,
            "error_type": bug.error_type,
            "error_message": bug.error_message,
            "confidence": bug.confidence,
        }), original_content)

        try:
            response = await self._llm.chat(prompt)
        except Exception:
            return None

        try:
            cleaned = _extract_json(response)
            raw = json.loads(cleaned)
            patched_content: str = raw.get("patched_content", original_content)
            diff: str = raw.get("diff", "")
        except (json.JSONDecodeError, TypeError, KeyError):
            patched_content = original_content
            diff = ""

        if not diff:
            diff = self._generate_diff(original_content, patched_content)

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(patched_content)
        except OSError:
            return PatchResult(
                file_path=bug.file_path,
                original_content=original_content,
                patched_content=patched_content,
                diff=diff,
            )

        return PatchResult(
            file_path=bug.file_path,
            original_content=original_content,
            patched_content=patched_content,
            diff=diff,
        )

    def _generate_diff(self, original: str, patched: str) -> str:
        original_lines = original.splitlines(keepends=True)
        patched_lines = patched.splitlines(keepends=True)
        return "".join(
            difflib.unified_diff(
                original_lines,
                patched_lines,
                fromfile="original",
                tofile="patched",
            )
        )
