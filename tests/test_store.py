"""Tests for RunStore: the persistence contract for Run records."""

from __future__ import annotations

import pytest

from mlcp.models import JobSpec, Run, RunResult, RunStatus
from mlcp.store import RunStore


@pytest.fixture
def store(tmp_path) -> RunStore:
    return RunStore(tmp_path / "test.db")


def _make_run(name: str = "job") -> Run:
    return Run(job=JobSpec(name=name, runner="linear_fit"))


def test_create_then_get_run_roundtrip(store: RunStore) -> None:
    run = _make_run()
    store.create_run(run)

    fetched = store.get_run(run.id)
    assert fetched is not None
    assert fetched.id == run.id
    assert fetched.status is RunStatus.PENDING
    assert fetched.job.name == run.job.name
    assert fetched.job.runner == "linear_fit"


def test_update_run_changes_status_and_document(store: RunStore) -> None:
    run = _make_run()
    store.create_run(run)

    run.status = RunStatus.SUCCEEDED
    run.attempts = 1
    run.result = RunResult(ok=True, outputs={"slope": 2.0}, metrics={"r2": 1.0})
    store.update_run(run)

    fetched = store.get_run(run.id)
    assert fetched is not None
    assert fetched.status is RunStatus.SUCCEEDED
    assert fetched.attempts == 1
    assert fetched.result is not None
    assert fetched.result.ok is True
    assert fetched.result.outputs["slope"] == 2.0


def test_list_runs_returns_newest_first(store: RunStore) -> None:
    from datetime import datetime, timedelta, timezone

    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    older = _make_run("older")
    older.created_at = base
    newer = _make_run("newer")
    newer.created_at = base + timedelta(seconds=10)

    # Insert older first so ordering can't be an artifact of insertion order.
    store.create_run(older)
    store.create_run(newer)

    runs = store.list_runs()
    assert [r.id for r in runs] == [newer.id, older.id]


def test_list_runs_filters_by_status(store: RunStore) -> None:
    succeeded = _make_run("ok")
    succeeded.status = RunStatus.SUCCEEDED
    failed = _make_run("bad")
    failed.status = RunStatus.FAILED

    store.create_run(succeeded)
    store.create_run(failed)

    only_succeeded = store.list_runs(status=RunStatus.SUCCEEDED)
    assert [r.id for r in only_succeeded] == [succeeded.id]

    only_failed = store.list_runs(status=RunStatus.FAILED)
    assert [r.id for r in only_failed] == [failed.id]


def test_get_run_missing_returns_none(store: RunStore) -> None:
    assert store.get_run("does-not-exist") is None


def test_update_run_unknown_raises_keyerror(store: RunStore) -> None:
    orphan = _make_run()  # never created in the store
    with pytest.raises(KeyError):
        store.update_run(orphan)
