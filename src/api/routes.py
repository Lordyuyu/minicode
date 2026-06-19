"""
API route handlers — task lifecycle + SSE streaming.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime

from fastapi import APIRouter, HTTPException, Request
from sse_starlette.sse import EventSourceResponse

from src.api.schemas import (
    PatchDetail,
    RepairRequest,
    RepairResponse,
    TaskResult,
    TaskStatus,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1")

# ---------------------------------------------------------------------------
# In-memory task store (replace with Postgres in production)
# ---------------------------------------------------------------------------

_tasks: dict[str, TaskStatus] = {}
_results: dict[str, TaskResult] = {}
_event_queues: dict[str, asyncio.Queue] = {}


def _create_task() -> str:
    task_id = str(uuid.uuid4())[:8]
    _tasks[task_id] = TaskStatus(
        task_id=task_id,
        status="pending",
        created_at=datetime.now().isoformat(),
    )
    _event_queues[task_id] = asyncio.Queue()
    return task_id


async def _emit_event(task_id: str, event: str, data: dict | None = None) -> None:
    """Push an SSE event into the task's queue."""
    payload = json.dumps({"event": event, "data": data or {}})
    q = _event_queues.get(task_id)
    if q:
        await q.put(payload)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/repair", response_model=RepairResponse, status_code=202)
async def start_repair(req: RepairRequest) -> RepairResponse:
    """Submit a new code-repair task.

    The task runs asynchronously.  Poll ``GET /repair/{task_id}`` for
    status or subscribe to ``GET /repair/{task_id}/stream`` for real-time
    SSE progress events.
    """
    task_id = _create_task()
    _tasks[task_id].status = "running"

    # Launch the pipeline in the background
    asyncio.create_task(_run_repair_pipeline(task_id, req))

    return RepairResponse(task_id=task_id)


@router.get("/repair/{task_id}", response_model=TaskResult | TaskStatus)
async def get_task_status(task_id: str):
    """Get the current status or final result of a repair task."""
    if task_id in _results:
        return _results[task_id]
    if task_id in _tasks:
        return _tasks[task_id]
    raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")


@router.get("/repair/{task_id}/stream")
async def stream_task(task_id: str, request: Request):
    """SSE endpoint — streams progress events for the given task.

    Connect with::

        curl -N http://localhost:8000/api/v1/repair/{task_id}/stream
    """
    if task_id not in _event_queues:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")

    q = _event_queues[task_id]

    async def event_generator():
        try:
            while True:
                # Check client disconnect
                if await request.is_disconnected():
                    break
                try:
                    payload = await asyncio.wait_for(q.get(), timeout=15.0)
                    yield {"event": "message", "data": payload}
                except TimeoutError:
                    # Send heartbeat to keep the connection alive
                    yield {"event": "heartbeat", "data": "{}"}
        except asyncio.CancelledError:
            pass

    return EventSourceResponse(event_generator())


# ---------------------------------------------------------------------------
# Pipeline runner (background task)
# ---------------------------------------------------------------------------


async def _run_repair_pipeline(task_id: str, req: RepairRequest) -> None:
    """Execute the MiniCode graph in the background, emitting SSE events."""
    try:
        from src.observability.tracer import setup_langsmith
        from src.orchestration.graph import MiniCodeGraph
        from src.utils.logger import setup_logging

        setup_logging()
        setup_langsmith()

        await _emit_event(task_id, "phase_change", {"phase": "starting"})

        graph = MiniCodeGraph()
        state = await graph.run(
            input_codebase_path=req.codebase_path,
            test_command=req.test_command,
        )

        # Emit progress events from metadata
        if state.bug_reports:
            await _emit_event(
                task_id,
                "bug_found",
                {"count": len(state.bug_reports)},
            )

        if state.patches:
            await _emit_event(
                task_id,
                "patch_generated",
                {"count": len(state.patches)},
            )

        await _emit_event(
            task_id,
            "complete",
            {
                "pipeline_success": state.pipeline_success,
                "patch_count": len(state.patches),
                "error_count": len(state.errors),
            },
        )

        # Build final result
        patches = [
            PatchDetail(
                file_path=p.file_path,
                diff=p.diff,
                verified=p.verified,
            )
            for p in state.patches
        ]

        _results[task_id] = TaskResult(
            task_id=task_id,
            status="completed" if state.pipeline_success else "failed",
            pipeline_success=state.pipeline_success,
            patches=patches,
            errors=state.errors,
            metadata=state.metadata,
        )
        _tasks[task_id].status = _results[task_id].status

    except Exception as exc:
        logger.exception("Pipeline {} failed", task_id)
        await _emit_event(task_id, "error", {"message": str(exc)})
        _results[task_id] = TaskResult(
            task_id=task_id,
            status="failed",
            pipeline_success=False,
            errors=[str(exc)],
        )
        _tasks[task_id].status = "failed"
