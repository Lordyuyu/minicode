from __future__ import annotations

import asyncio
import difflib
import json
import os
import shutil

from src.core.state import AgentState
from src.core.types import BugReport, PatchResult
from src.llm.deepseek_client import DeepSeekClient
from src.llm.prompt_templates import build_patch_generation_prompt
from src.utils.logger import get_logger
from src.utils.text import extract_json

logger = get_logger(__name__)


class PatchGenerator:
    def __init__(self, llm_client: DeepSeekClient) -> None:
        self._llm = llm_client

    async def generate(self, state: AgentState) -> list[PatchResult]:
        """Generate patches sequentially for each bug report."""
        patches: list[PatchResult] = []
        for bug in state.bug_reports:
            result = await self._generate_for_bug(bug, state.input_codebase_path)
            if result is not None:
                patches.append(result)
        state.patches = patches
        return patches

    async def generate_concurrent(self, state: AgentState) -> list[PatchResult]:
        """Generate patches **concurrently** for all bug reports.

        Each bug is processed independently via ``asyncio.gather``.
        Failures in one bug do not block others — they return ``None``
        and are filtered out.
        """
        if not state.bug_reports:
            state.patches = []
            return []

        logger.info(
            "Generating patches for {} bug(s) concurrently",
            len(state.bug_reports),
        )
        results = await asyncio.gather(
            *(
                self._generate_for_bug(bug, state.input_codebase_path)
                for bug in state.bug_reports
            ),
            return_exceptions=True,
        )

        patches: list[PatchResult] = []
        for bug, result in zip(state.bug_reports, results):
            if isinstance(result, Exception):
                logger.error(
                    "Patch generation failed for {} ({}:{}-{}): {}",
                    bug.file_path,
                    bug.line_start,
                    bug.line_end,
                    result,
                )
                continue
            if result is not None:
                patches.append(result)

        state.patches = patches
        logger.info(
            "Generated {} patch(es) ({} failed)",
            len(patches),
            len(state.bug_reports) - len(patches),
        )
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
            logger.warning("Cannot read file for patching: {}", file_path)
            return None

        prompt = build_patch_generation_prompt(
            json.dumps(
                {
                    "file_path": bug.file_path,
                    "line_start": bug.line_start,
                    "line_end": bug.line_end,
                    "error_type": bug.error_type,
                    "error_message": bug.error_message,
                    "confidence": bug.confidence,
                }
            ),
            original_content,
        )

        try:
            response = await self._llm.chat(prompt)
        except Exception:
            logger.exception("LLM call failed for bug at {}", bug.file_path)
            return None

        try:
            cleaned = extract_json(response)
            raw = json.loads(cleaned)
            patched_content: str = raw.get("patched_content", original_content)
            diff: str = raw.get("diff", "")
        except (json.JSONDecodeError, TypeError, KeyError):
            logger.warning(
                "Failed to parse LLM patch response as JSON for {}", bug.file_path
            )
            patched_content = original_content
            diff = ""

        if not diff:
            diff = self._generate_diff(original_content, patched_content)

        # Backup original before overwriting
        backup_path = file_path + ".minicode.bak"
        try:
            shutil.copy2(file_path, backup_path)
        except OSError:
            backup_path = ""  # best-effort, proceed without backup

        try:
            os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(patched_content)
        except OSError:
            logger.exception("Failed to write patch to {}", file_path)
            return PatchResult(
                file_path=bug.file_path,
                original_content=original_content,
                patched_content=patched_content,
                diff=diff,
            )

        logger.info(
            "Applied patch to {} (backup: {})",
            file_path,
            backup_path or "none",
        )
        return PatchResult(
            file_path=bug.file_path,
            original_content=original_content,
            patched_content=patched_content,
            diff=diff,
        )

    @staticmethod
    def _generate_diff(original: str, patched: str) -> str:
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
