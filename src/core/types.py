from __future__ import annotations

from enum import StrEnum, auto
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4


class AgentPhase(StrEnum):
    INTENT_RECOGNITION = auto()
    PLANNING = auto()
    EXECUTION = auto()
    VERIFICATION = auto()


class SkillCategory(StrEnum):
    BUG_LOCALIZATION = auto()
    PATCH_GENERATION = auto()
    TEST_VERIFICATION = auto()
    CODE_ANALYSIS = auto()
    DEPENDENCY_RESOLUTION = auto()


class MemoryType(StrEnum):
    EPISODIC = auto()
    PROCEDURAL = auto()
    SEMANTIC = auto()


@dataclass
class Skill:
    skill_id: str = field(default_factory=lambda: str(uuid4()))
    name: str = ""
    category: SkillCategory = SkillCategory.CODE_ANALYSIS
    description: str = ""
    embedding: list[float] = field(default_factory=list)
    success_rate: float = 0.0
    invocation_count: int = 0


@dataclass
class ContextChunk:
    chunk_id: str = field(default_factory=lambda: str(uuid4()))
    content: str = ""
    summary: str = ""
    token_count: int = 0
    compressed: bool = False
    storage_key: str = ""


@dataclass
class MemoryEntry:
    memory_id: str = field(default_factory=lambda: str(uuid4()))
    memory_type: MemoryType = MemoryType.EPISODIC
    task_description: str = ""
    actions: list[dict[str, Any]] = field(default_factory=list)
    outcome: str = ""
    embedding: list[float] = field(default_factory=list)
    procedural_pattern: str = ""
    timestamp: float = 0.0


@dataclass
class PatchResult:
    file_path: str = ""
    original_content: str = ""
    patched_content: str = ""
    diff: str = ""
    verified: bool = False
    verification_output: str = ""


@dataclass
class BugReport:
    file_path: str = ""
    line_start: int = 0
    line_end: int = 0
    error_type: str = ""
    error_message: str = ""
    confidence: float = 0.0


class PermissionScope(StrEnum):
    READ_ONLY = auto()
    CODE_WRITE = auto()
    TEST_EXEC = auto()
    FULL_ACCESS = auto()


@dataclass
class SubAgentTask:
    agent_id: str = field(default_factory=lambda: str(uuid4()))
    agent_type: str = ""
    task_description: str = ""
    permission: PermissionScope = PermissionScope.READ_ONLY
    target_files: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    result: str = ""
    success: bool = False


@dataclass
class SkillV2(Skill):
    tags: list[str] = field(default_factory=list)
    applicability: str = ""
    examples: list[str] = field(default_factory=list)
    boundaries: str = ""
    directory: str = ""


@dataclass
class PromptCacheEntry:
    cache_key: str = ""
    content_hash: str = ""
    summary: str = ""
    token_count: int = 0
    ttl: int = 300
    hit_count: int = 0
