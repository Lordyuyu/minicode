from __future__ import annotations

import json
import re
from typing import Any


class PatchValidator:
    SENSITIVE_PATTERNS: list[str] = [
        r"api[_-]?key\s*[:=]\s*[\"'].+?[\"']",
        r"secret\s*[:=]\s*[\"'].+?[\"']",
        r"token\s*[:=]\s*[\"'].+?[\"']",
        r"password\s*[:=]\s*[\"'].+?[\"']",
        r"subprocess\.(?:call|Popen|run|check_output|check_call)\s*\(",
        r"\beval\s*\(",
        r"\bexec\s*\(",
        r"os\.system\s*\(",
        r"os\.popen\s*\(",
        r"base64\.(?:b64decode|decode|decodestring)\s*\(",
        r"__import__\s*\(",
        r"\bcompile\s*\(",
        r"pickle\.loads\s*\(",
    ]

    def __init__(self, patterns: list[str] | None = None) -> None:
        self._patterns = patterns if patterns is not None else self.SENSITIVE_PATTERNS
        self._compiled = [re.compile(p, re.IGNORECASE) for p in self._patterns]

    def validate(self, patch_content: str) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        for raw_pattern, compiled in zip(self._patterns, self._compiled):
            if compiled.search(patch_content):
                reasons.append(f"Matched sensitive pattern: {raw_pattern}")
        return (len(reasons) == 0, reasons)


class AIRiskClassifier:
    def __init__(self, llm_client: Any) -> None:
        self._llm = llm_client

    async def classify(self, patch_content: str) -> dict[str, Any]:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a security reviewer. Analyze the following code patch "
                    "and classify its risk. Respond in JSON with keys: "
                    "risk_level (one of low/medium/high), "
                    "risk_type (one of prompt_injection/malicious_code/secret_leak/"
                    "dangerous_operation/safe), "
                    "reasoning (short explanation)."
                ),
            },
            {"role": "user", "content": f"Review this patch:\n\n{patch_content}"},
        ]
        response = await self._llm.chat(messages)
        try:
            result = json.loads(response)
        except json.JSONDecodeError:
            result = {
                "risk_level": "high",
                "risk_type": "unknown",
                "reasoning": "Failed to parse LLM response as JSON",
            }
        return result
