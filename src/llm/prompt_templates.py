from __future__ import annotations

from typing import Any


def build_intent_recognition_prompt(task_description: str, error_output: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": (
                "You are an intent recognition engine for an automated code repair system. "
                "Analyze the task description and error output to determine the user's intent. "
                "Classify the intent into one of: bug_fix, feature_add, refactor, dependency_fix. "
                "Output ONLY a JSON object with keys: 'intent', 'confidence', and 'reasoning'."
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
    error_output: str, codebase_structure: str, relevant_files: list[tuple[str, str]]
) -> list[dict[str, Any]]:
    file_snippets = "\n---\n".join(
        [f"File: {path}\n```python\n{content}\n```" for path, content in relevant_files]
    )
    return [
        {
            "role": "system",
            "content": (
                "You are a bug localization expert. Given test error output and source code, "
                "identify the exact file, line range, and root cause of each bug. "
                "Output a JSON array of bugs with: file_path, line_start, line_end, error_type, error_message, confidence."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Error output:\n```\n{error_output}\n```\n\n"
                f"Codebase structure:\n{codebase_structure}\n\n"
                f"Relevant files:\n{file_snippets}\n\n"
                "Localize the bugs."
            ),
        },
    ]


def build_patch_generation_prompt(
    bug_report: str, file_content: str
) -> list[dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": (
                "You are a patch generation expert. Given a bug report and the source file content, "
                "generate a minimal, correct patch. Output valid JSON with: 'file_path', 'original_content', "
                "'patched_content', 'diff' (unified diff format)."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Bug report:\n{bug_report}\n\n"
                f"File content:\n```python\n{file_content}\n```\n\n"
                "Generate the patch."
            ),
        },
    ]


def build_skill_ranking_prompt(
    task_description: str, candidates: list[dict[str, Any]]
) -> str:
    candidates_text = "\n".join(
        [
            f"{i+1}. {c['name']} ({c['category']}): {c['description']} [similarity={c.get('similarity', 0):.3f}]"
            for i, c in enumerate(candidates)
        ]
    )
    return (
        f"Given the task: '{task_description}'\n\n"
        f"Rank the following skill candidates by relevance (pick the single best one):\n"
        f"{candidates_text}\n\n"
        "Output ONLY the number (1-based) of the best skill, nothing else."
    )
