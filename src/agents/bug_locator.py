from __future__ import annotations

import json
import os
from pathlib import Path

from src.core.state import AgentState
from src.core.types import BugReport
from src.llm.deepseek_client import DeepSeekClient
from src.llm.prompt_templates import build_bug_localization_prompt
from src.utils.text import extract_json


class BugLocator:
    def __init__(self, llm_client: DeepSeekClient) -> None:
        self._llm = llm_client

    async def localize(self, state: AgentState) -> list[BugReport]:
        test_output = state.metadata.get("test_output", "")
        codebase_path = state.input_codebase_path
        codebase_structure = self._get_codebase_structure(codebase_path)
        relevant_files = self._find_relevant_files(codebase_path, test_output)

        prompt = build_bug_localization_prompt(test_output, codebase_structure, relevant_files)
        try:
            response = await self._llm.chat(prompt)
            cleaned = extract_json(response)
            raw = json.loads(cleaned)
            reports = [BugReport(**item) for item in raw]
        except Exception:
            import traceback

            reports = []
            state.errors.append(f"BugLocator: {traceback.format_exc()}")

        state.bug_reports = reports
        return reports

    def _get_codebase_structure(self, path: str) -> str:
        lines_out: list[str] = []
        root = Path(path)
        for entry in sorted(root.rglob("*")):
            if entry.is_file() and entry.suffix == ".py":
                relative = entry.relative_to(root)
                lines_out.append(str(relative))
        return "\n".join(lines_out)

    def _find_relevant_files(self, path: str, test_output: str) -> list[tuple[str, str]]:
        candidates: list[tuple[str, str]] = []
        seen: set[str] = set()

        for line in test_output.splitlines():
            line = line.strip()
            if "File" in line and ".py" in line:
                extracted = [w for w in line.replace('"', " ").replace("'", " ").split() if w.endswith(".py")]
                for match in extracted:
                    # Sanitize: resolve relative paths safely under codebase
                    raw_path = os.path.normpath(os.path.join(path, match))
                    if not raw_path.startswith(os.path.normpath(path) + os.sep) and raw_path != os.path.normpath(path):
                        continue  # path traversal attempt
                    abs_path = raw_path
                    if os.path.isfile(abs_path) and abs_path not in seen:
                        seen.add(abs_path)
                        try:
                            with open(abs_path, encoding="utf-8") as f:
                                candidates.append((abs_path, f.read()))
                        except OSError:
                            continue

        if not candidates:
            root = Path(path)
            for entry in sorted(root.rglob("*.py")):
                if str(entry) not in seen:
                    seen.add(str(entry))
                    try:
                        candidates.append((str(entry), entry.read_text(encoding="utf-8")))
                    except OSError:
                        continue

        return candidates
