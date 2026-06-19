from __future__ import annotations

import json
import re
from typing import Any

from src.utils.logger import get_logger

logger = get_logger(__name__)


class PatchValidator:
    """Regex-based static security scanner for code patches.

    Patterns are compiled with ``re.IGNORECASE | re.DOTALL`` so that
    multi-line constructs (e.g. ``eval(\\n  "code"\\n)``) are detected.
    """

    SENSITIVE_PATTERNS: list[str] = [
        # -- credential / secret leakage --
        r"api[_-]?key\s*[:=]\s*[\"'].+?[\"']",
        r"secret\s*[:=]\s*[\"'].+?[\"']",
        r"token\s*[:=]\s*[\"'].+?[\"']",
        r"password\s*[:=]\s*[\"'].+?[\"']",
        # -- subprocess / command execution --
        r"subprocess\.(?:call|Popen|run|check_output|check_call|getoutput|getstatusoutput)\s*\(",
        r"os\.system\s*\(",
        r"os\.popen\s*\(",
        r"os\.exec(?:v[pe]?[e]?|l[pe]?[e]?)\s*\(",
        r"pty\.spawn\s*\(",
        # -- code execution --
        r"\beval\s*\(",
        r"\bexec\s*\(",
        r"\bcompile\s*\(",
        r"__import__\s*\(",
        # -- dangerous pickling / serialisation --
        r"pickle\.(?:loads?|dump)\s*\(",
        # -- base64 and encoding-based obfuscation --
        r"base64\.(?:b64decode|decode|decodestring|standard_b64decode|urlsafe_b64decode)\s*\(",
        r"codecs\.decode\s*\(",
        # -- raw file writes (could overwrite system files) --
        r"\bopen\s*\([^)]*[\"'][wab]",
        r"pathlib\.Path\s*\([^)]*\)\s*\.\s*write_(?:text|bytes)\s*\(",
        # -- ctypes / FFI (arbitrary native code execution) --
        r"ctypes\.(?:CDLL|WinDLL|OleDLL|cdll|windll|oledll|pydll|PYDLL|LibraryLoader)\s*\(",
        r"ctypes\.(?:c_|py_object|POINTER|cast|addressof|string_at)\b",
        # -- introspection / sandbox-escape patterns --
        r"__subclasses__\s*\(\s*\)",
        r"__globals__\s*\[",
        r"__builtins__\s*\[",
        r"__reduce__\s*\(",
        # -- network access (exfiltration vector) --
        r"(?:requests|urllib\.request|urllib3|httpx|aiohttp)\.(?:get|post|put|delete|patch|request)\s*\(",
        r"socket\.(?:socket|create_connection|connect)\s*\(",
        # -- filesystem destruction --
        r"shutil\.rmtree\s*\(",
        r"os\.remove\s*\(",
        r"os\.unlink\s*\(",
        r"os\.rmdir\s*\(",
        r"os\.chmod\s*\(",
        # -- reverse shell patterns --
        r"/dev/tcp/",
        r"/dev/udp/",
    ]

    def __init__(self, patterns: list[str] | None = None) -> None:
        self._patterns = patterns if patterns is not None else self.SENSITIVE_PATTERNS
        # Compile with IGNORECASE + DOTALL for multi-line detection
        self._compiled = [
            re.compile(p, re.IGNORECASE | re.DOTALL) for p in self._patterns
        ]

    def validate(self, patch_content: str) -> tuple[bool, list[str]]:
        """Scan *patch_content* against all patterns.

        Returns:
            ``(is_safe, reasons)`` where *is_safe* is ``True`` when no
            pattern matched, and *reasons* lists the patterns that fired.
        """
        reasons: list[str] = []
        for raw_pattern, compiled in zip(self._patterns, self._compiled):
            try:
                if compiled.search(patch_content):
                    reasons.append(f"Matched sensitive pattern: {raw_pattern}")
            except re.error:
                logger.warning("Invalid regex pattern skipped: {}", raw_pattern)
        return (len(reasons) == 0, reasons)


class AIRiskClassifier:
    def __init__(self, llm_client: Any) -> None:
        self._llm = llm_client

    async def classify(self, patch_content: str) -> dict[str, Any]:
        """Use the LLM to classify the risk level of a code patch.

        On JSON parse failure the default is ``risk_level: "medium"``
        (not ``"high"``) to avoid blocking legitimate patches during
        temporary API degradation.
        """
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
        try:
            response = await self._llm.chat(messages)
        except Exception:
            logger.exception("AI risk classification call failed")
            return {
                "risk_level": "medium",
                "risk_type": "api_error",
                "reasoning": "LLM call failed — defaulting to medium risk",
            }

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            logger.warning("Failed to parse AI classifier response as JSON: {!r}", response[:200])
            return {
                "risk_level": "medium",
                "risk_type": "parse_error",
                "reasoning": "Failed to parse LLM response as JSON",
            }
