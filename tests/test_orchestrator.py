"""Tests for the Orchestrator: lifecycle ownership, retries, and durability."""

from __future__ import annotations

from typing import Any

import pytest

from mlcp.models import JobSpec, RunResult, RunStatus
from mlcp.orchestrator import Orchestrator
from mlcp.runner import Runner, RunnerRegistry
from mlcp.runners.builtin import FailingRunner, LinearFitRunner
from mlcp.store import RunStore


@pytest.fixture
def store(tmp_path) -> RunStore:
    return RunStore(tmp_path / "test.db")


@pytest.fixture
def registry() -> RunnerRegistry:
    reg = RunnerRegistry()
    reg.register(LinearFitRunner())
    reg.register(FailingRunner())
    return reg


@pytest.fixture
def orchestrator(store: RunStore, registry: RunnerRegistry) -> Orchestrator:
    return Orchestrator(store, registry, sync=True)


def test_submit_linear_fit_succeeds(orchestrator: Orchestrator) -> None:
    job = JobSpec(
        name="fit",
        runner="linear_fit",
        params={"x": [1, 2, 3, 4], "y": [2, 4, 6, 8]},
    )
    run = orchestrator.submit(job)

    assert run.status is RunStatus.SUCCEEDED
    assert run.attempts == 1
    assert run.result is not None
    assert run.result.ok is True
    assert run.result.metrics["r2"] == pytest.approx(1.0)
    assert run.result.outputs["slope"] == pytest.approx(2.0)


def test_submit_always_fail_exhausts_retries(orchestrator: Orchestrator) -> None:
    job = JobSpec(name="boom", runner="always_fail", max_retries=2)
    run = orchestrator.submit(job)

    assert run.status is RunStatus.FAILED
    assert run.attempts == 3  # initial attempt + 2 retries
    assert run.result is not None
    assert run.result.ok is False
    assert run.result.error  # error is set / non-empty


def test_submit_flaky_runner_recovers_on_retry(
    store: RunStore, registry: RunnerRegistry
) -> None:
    class FlakyRunner(Runner):
        name = "flaky"

        def __init__(self) -> None:
            self.calls = 0

        def run(self, params: dict[str, Any]) -> RunResult:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("transient failure")
            return RunResult(ok=True, outputs={"recovered": True})

    registry.register(FlakyRunner())
    orchestrator = Orchestrator(store, registry, sync=True)

    job = JobSpec(name="flaky-job", runner="flaky", max_retries=1)
    run = orchestrator.submit(job)

    assert run.status is RunStatus.SUCCEEDED
    assert run.attempts == 2
    assert run.result is not None
    assert run.result.ok is True


def test_submit_unknown_runner_raises_and_persists_nothing(
    orchestrator: Orchestrator, store: RunStore
) -> None:
    job = JobSpec(name="ghost", runner="does_not_exist")
    with pytest.raises(KeyError):
        orchestrator.submit(job)

    # Nothing should have been written for a fast-failed submission.
    assert store.list_runs() == []
