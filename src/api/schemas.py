"""Pydantic models for the MiniCode REST API."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

# -- Request -----------------------------------------------------------------

class RepairRequest(BaseModel):
    """Request body for starting a code repair task."""

    codebase_path: str = Field(
        ...,
        description="Absolute path to the Python codebase",
        examples=["/home/user/projects/myapp"],
    )
    test_command: str = Field(
        default="pytest",
        description="Test command to run (default: pytest)",
        examples=["pytest tests/ -x"],
    )


# -- Response ----------------------------------------------------------------

class RepairResponse(BaseModel):
    """Response returned when a repair task is created."""

    task_id: str
    status: str = "accepted"


class TaskStatus(BaseModel):
    """Current status of a repair task."""

    task_id: str
    status: str  # pending | running | completed | failed
    current_phase: str | None = None
    bug_count: int = 0
    patch_count: int = 0
    verified_count: int = 0
    errors: list[str] = []
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class PatchDetail(BaseModel):
    file_path: str
    diff: str
    verified: bool


class TaskResult(BaseModel):
    """Final result of a completed repair task."""

    task_id: str
    status: str  # completed | failed
    pipeline_success: bool
    patches: list[PatchDetail] = []
    errors: list[str] = []
    metadata: dict[str, Any] = {}


# -- SSE events --------------------------------------------------------------

class SSEEvent(BaseModel):
    """A single event emitted over the SSE stream."""

    event: str
    """Event type: phase_change, bug_found, patch_generated, verification,
    safety_review, error, complete."""

    data: dict[str, Any] = {}
    """Event payload as a free-form dict."""

    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
