"""Core domain models. Everything that crosses a boundary (API, store, runner)
is one of these, so the contract lives in exactly one place."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return uuid4().hex


class RunStatus(str, Enum):
    """Lifecycle states for a Run. Owned by the orchestrator, never by a runner."""

    PENDING = "pending"
    RUNNING = "running"
    RETRYING = "retrying"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self in {RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED}


class JobSpec(BaseModel):
    """What to run. Supplied by a caller; treated as immutable once accepted."""

    name: str = Field(..., description="Human-readable label for the job.")
    runner: str = Field(..., description="Key of a registered runner.")
    params: dict[str, Any] = Field(default_factory=dict, description="Runner inputs.")
    max_retries: int = Field(
        default=0, ge=0, description="Retries attempted after the first failure."
    )
    interval_seconds: Optional[int] = Field(
        default=None,
        gt=0,
        description="If set, the orchestrator schedules this job to recur at this interval.",
    )


class RunResult(BaseModel):
    """The outcome a runner reports (or that the orchestrator synthesizes on error)."""

    ok: bool
    outputs: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict)
    error: Optional[str] = None


class Run(BaseModel):
    """A single execution of a JobSpec. The persisted Run record IS the audit
    trail in M1 — every state transition is written through the store."""

    id: str = Field(default_factory=_new_id)
    job: JobSpec
    status: RunStatus = RunStatus.PENDING
    attempts: int = 0
    created_at: datetime = Field(default_factory=_utcnow)
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    result: Optional[RunResult] = None
