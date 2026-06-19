"""
Text processing utilities shared across agent modules.

Provides a single, robust JSON extraction function used by BugLocator,
PatchGenerator, and any other module that needs to parse LLM JSON output.
"""

from __future__ import annotations

import re


def extract_json(text: str) -> str:
    """Extract JSON content from LLM response text.

    Handles multiple common LLM output formats:
    - Triple-backtick fenced code blocks with optional ``json`` language tag
    - Raw JSON without fences
    - Single-backtick fenced blocks (less common, handled as fallback)

    Returns the extracted JSON string, stripped of surrounding whitespace
    and code fence markers. If no fences are found, returns the original
    text stripped.
    """
    # Primary: triple-backtick fences (most LLMs use this)
    match = re.search(r'```(?:json)?\s*\n(.*?)\n\s*```', text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Fallback: single-backtick fences (rare, some older models)
    match = re.search(r'`(?:json)?\s*\n(.*?)\n\s*`', text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # No fences found — assume raw JSON
    return text.strip()
