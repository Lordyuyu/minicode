from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from src.core.types import (
    AgentPhase, Skill, ContextChunk, MemoryEntry,
    PatchResult, BugReport,
)


@dataclass
class AgentState:
    task_id: str = ""
    input_codebase_path: str = ""
    test_command: str = ""

    current_phase: AgentPhase = AgentPhase.INTENT_RECOGNITION
    messages: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    identified_intent: str = ""
    selected_skill: Skill | None = None
    skill_candidates: list[Skill] = field(default_factory=list)

    execution_plan: list[str] = field(default_factory=list)
    current_step_index: int = 0

    context_chunks: list[ContextChunk] = field(default_factory=list)
    compressed_context: str = ""

    bug_reports: list[BugReport] = field(default_factory=list)
    patches: list[PatchResult] = field(default_factory=list)

    verification_results: list[dict[str, Any]] = field(default_factory=list)
    pipeline_success: bool = False

    human_review_required: bool = False
    human_approved: bool = False

    memory_entries: list[MemoryEntry] = field(default_factory=list)

    metadata: dict[str, Any] = field(default_factory=dict)
    trace_id: str = ""
