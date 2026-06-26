from __future__ import annotations

from typing import Any


def build_intent_recognition_prompt(
    task_description: str, error_output: str
) -> list[dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": (
                "You are an intent recognition engine for an automated code repair system. "
                "Analyze the task description and error output to determine the user's intent. "
                "Classify the intent into one of: bug_fix, feature_add, refactor, dependency_fix.\n\n"
                "Examples:\n"
                '- "Fix the division by zero error" → intent: bug_fix, confidence: 0.95\n'
                '- "Add a new sqrt function" → intent: feature_add, confidence: 0.90\n'
                '- "Clean up the utils module" → intent: refactor, confidence: 0.85\n'
                '- "Tests fail after upgrading requests" → intent: dependency_fix, confidence: 0.80\n\n'
                "Output ONLY a JSON object with keys: 'intent', 'confidence', and 'reasoning'.\n"
                "The 'reasoning' field should contain a brief explanation of your classification."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Task: {task_description}\n"
                f"Error output from test run:\n```\n{error_output}\n```\n"
                "Determine the intent."
            ),
        },
    ]


def build_bug_localization_prompt(
    error_output: str,
    codebase_structure: str,
    relevant_files: list[tuple[str, str]],
) -> list[dict[str, Any]]:
    file_snippets = "\n---\n".join(
        [
            f"File: {path}\n```python\n{content}\n```"
            for path, content in relevant_files
        ]
    )

    few_shot = (
        "Example 1:\n"
        'Error: "ZeroDivisionError: division by zero" in test_divide\n'
        "File: calculator.py contains:\n"
        "```python\n"
        "def divide(a, b):\n"
        "    return a / b\n"
        "```\n"
        "Output:\n"
        "```json\n"
        '[{"file_path": "calculator.py", "line_start": 1, "line_end": 2, '
        '"error_type": "ZeroDivisionError", '
        '"error_message": "division by zero when b=0", "confidence": 0.92}]\n'
        "```\n"
        "\n"
        "Example 2:\n"
        'Error: "IndexError: list index out of range" at utils.py:15\n'
        "File: utils.py line 15 is `return items[idx]`\n"
        "Output:\n"
        "```json\n"
        '[{"file_path": "utils.py", "line_start": 14, "line_end": 16, '
        '"error_type": "IndexError", '
        '"error_message": "No bounds check before list access", '
        '"confidence": 0.88}]\n'
        "```\n"
        "\n"
        "Example 3:\n"
        'Error: "KeyError: \'timeout\'" at config/reader.py:5\n'
        "File: config/reader.py line 5 is `return DEFAULTS[key]`\n"
        "Output:\n"
        "```json\n"
        '[{"file_path": "config/reader.py", "line_start": 4, "line_end": 5, '
        '"error_type": "KeyError", '
        '"error_message": "Dict access without safe .get() fallback", '
        '"confidence": 0.93}]\n'
        "```\n"
        "\n"
        "Example 4:\n"
        'Error: "AssertionError: assert default_val == override_val" in '
        "test_deep_merge. Test expects override values to win, but default "
        "values are returned instead.\n"
        "File: merger.py contains a deep_merge that compares isinstance(dict) "
        "for both sides, but the else branch returns the wrong source:\n"
        "```python\n"
        "def deep_merge(default, override):\n"
        "    result = {}\n"
        "    for key in default:\n"
        "        if key in override:\n"
        "            if isinstance(default[key], dict) and isinstance(override[key], dict):\n"
        "                result[key] = deep_merge(default[key], override[key])\n"
        "            else:\n"
        "                result[key] = default[key]  # logically wrong\n"
        "```\n"
        "Output:\n"
        "```json\n"
        '[{"file_path": "merger.py", "line_start": 6, "line_end": 7, '
        '"error_type": "LogicError", '
        '"error_message": "else branch returns default[key] instead of '
        'override[key] when values are not both dicts", '
        '"confidence": 0.90}]\n'
        "```\n"
        "\n"
        "Example 5:\n"
        'Error: "AssertionError: assert is_valid(-1) is False" — negative '
        "numbers should be rejected, but the function returns True.\n"
        "File: validator.py contains:\n"
        "```python\n"
        "def is_valid(x):\n"
        "    if x > 0:\n"
        "        return True\n"
        "    return True  # should be False for non-positive\n"
        "```\n"
        "Output:\n"
        "```json\n"
        '[{"file_path": "validator.py", "line_start": 3, "line_end": 4, '
        '"error_type": "LogicError", '
        '"error_message": "Default return is True instead of False, '
        'so negative numbers pass validation", '
        '"confidence": 0.92}]\n'
        "```\n"
        "\n"
        "Example 6:\n"
        'Error: "AssertionError: assert result == [1, 2]" — test expects 2 '
        "elements but function returns 3.\n"
        "File: processor.py contains:\n"
        "```python\n"
        "def take_first_n(items, n):\n"
        "    return items[:n+1]  # off-by-one: should be items[:n]\n"
        "```\n"
        "Output:\n"
        "```json\n"
        '[{"file_path": "processor.py", "line_start": 1, "line_end": 2, '
        '"error_type": "LogicError", '
        '"error_message": "Slice uses n+1 instead of n, returning one '
        'extra element", '
        '"confidence": 0.94}]\n'
        "```"
    )

    return [
        {
            "role": "system",
            "content": (
                "You are a bug localization expert. Given test error output and source code, "
                "identify the exact file, line range, and root cause of each bug.\n\n"
                "IMPORTANT: Always report the SOURCE file that contains the bug, "
                "NOT the test file where the assertion failed.\n\n"
                "Think step by step:\n"
                "1. Parse the error output to identify the exception type and traceback.\n"
                "2. Map each traceback entry to the corresponding source file (skip test files).\n"
                "3. For each bug, determine the narrowest line range that captures the root cause.\n"
                "4. Estimate confidence based on how clearly the traceback points to the cause.\n\n"
                + few_shot
                + "\n\n"
                "Output a JSON array of bugs with: file_path, line_start, line_end, "
                "error_type, error_message, confidence."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Error output:\n```\n{error_output}\n```\n\n"
                f"Codebase structure:\n{codebase_structure}\n\n"
                f"Relevant files:\n{file_snippets}\n\n"
                "Localize the bugs. Follow the chain-of-thought steps above."
            ),
        },
    ]


def build_patch_generation_prompt(
    bug_report: str, file_content: str
) -> list[dict[str, Any]]:
    few_shot = """Example 1 — ZeroDivisionError fix:
Bug: divide(a, b) raises ZeroDivisionError when b=0
Original:
```python
def divide(a, b):
    return a / b
```
Patch:
```python
def divide(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
```

Example 2 — IndexError fix:
Bug: get_item(items, idx) raises IndexError when idx >= len(items)
Original:
```python
def get_item(items, idx):
    return items[idx]
```
Patch:
```python
def get_item(items, idx):
    if idx < 0 or idx >= len(items):
        raise IndexError(f"Index {idx} out of range [0, {len(items)})")
    return items[idx]
```

Example 3 — KeyError / dict access fix:
Bug: get_config(key) raises KeyError when key is missing; test expects None
Original:
```python
DEFAULTS = {'host': 'localhost', 'port': 8080}

def get_config(key: str):
    return DEFAULTS[key]
```
Patch:
```python
DEFAULTS = {'host': 'localhost', 'port': 8080}

def get_config(key: str):
    return DEFAULTS.get(key)
```

Example 4 — Mutable default argument fix:
Bug: append_item uses target=[] (mutable default), tests fail because all calls share the same list
Original:
```python
def append_item(item, target=[]):
    target.append(item)
    return target
```
Patch:
```python
def append_item(item, target=None):
    if target is None:
        target = []
    target.append(item)
    return target
```

Example 5 — Inverted condition / wrong default return fix:
Bug: is_valid(x) returns True for negative numbers; test expects False when x <= 0
Original:
```python
def is_valid(x):
    if x > 0:
        return True
    return True  # should be False
```
Patch:
```python
def is_valid(x):
    if x > 0:
        return True
    return False
```

Example 6 — Off-by-one / wrong slice boundary fix:
Bug: take_first_n(items, n) returns n+1 elements instead of n; test expects exactly n elements
Original:
```python
def take_first_n(items, n):
    return items[:n+1]  # off-by-one
```
Patch:
```python
def take_first_n(items, n):
    return items[:n]
```

Example 7 — Wrong branch returns default instead of override fix:
Bug: deep_merge returns default[key] in the else branch; test expects override[key] to win
Original:
```python
def deep_merge(default, override):
    result = {}
    for key in default:
        if key in override:
            if isinstance(default[key], dict) and isinstance(override[key], dict):
                result[key] = deep_merge(default[key], override[key])
            else:
                result[key] = default[key]  # BUG
    return result
```
Patch:
```python
def deep_merge(default, override):
    result = {}
    for key in default:
        if key in override:
            if isinstance(default[key], dict) and isinstance(override[key], dict):
                result[key] = deep_merge(default[key], override[key])
            else:
                result[key] = override[key]
    return result
```"""

    return [
        {
            "role": "system",
            "content": (
                "You are a patch generation expert. Given a bug report and the source file content, "
                "generate a MINIMAL, correct patch.\n\n"
                "CRITICAL rules:\n"
                "- Only change the SOURCE file that contains the bug. NEVER modify test files.\n"
                "- Make the SMALLEST possible change — do NOT add new dictionary keys, "
                "do NOT add new imports, do NOT rename parameters unnecessarily.\n"
                "- Keep the function signature identical UNLESS the bug IS the signature "
                "(e.g. mutable default arg target=[] → target=None).\n"
                "- Study what the failing tests EXPECT, and make the code match that expectation.\n"
                "- If a KeyError occurs, use .get() — do NOT add the missing key to the dict.\n\n"
                "Think step by step:\n"
                "1. Understand what the test expects vs what the buggy code does.\n"
                "2. Identify the ONE minimal change that bridges the gap.\n"
                "3. Apply ONLY that change — resist improving anything else.\n"
                "4. Write the FULL patched file content.\n"
                "5. Generate a unified diff.\n\n"
                + few_shot
                + "\n\n"
                "Output valid JSON with keys:\n"
                "- 'file_path': the path of the modified SOURCE file (NOT the test file)\n"
                "- 'original_content': the file content before patching\n"
                "- 'patched_content': the COMPLETE file content after patching\n"
                "- 'diff': unified diff format showing exactly what changed"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Bug report:\n{bug_report}\n\n"
                f"File content:\n```python\n{file_content}\n```\n\n"
                "Generate the patch. Output ONLY valid JSON."
            ),
        },
    ]


def build_skill_ranking_prompt(
    task_description: str, candidates: list[dict[str, Any]]
) -> str:
    candidates_text = "\n".join(
        [
            f"{i+1}. {c['name']} ({c['category']}): {c['description']} "
            f"[similarity={c.get('similarity', 0):.3f}]"
            for i, c in enumerate(candidates)
        ]
    )
    return (
        f"Given the task: '{task_description}'\n\n"
        f"Rank the following skill candidates by relevance.\n\n"
        f"{candidates_text}\n\n"
        "Think step by step:\n"
        "1. Compare the task description against each candidate's category and description.\n"
        "2. Consider both semantic similarity and practical applicability.\n"
        "3. If two candidates are close, prefer the one with higher past success rate.\n"
        "4. Select the SINGLE best match.\n\n"
        "Output ONLY the number (1-based) of the best skill, nothing else."
    )
