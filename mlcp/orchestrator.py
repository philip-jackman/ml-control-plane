"""The orchestrator: the one component that owns a run's lifecycle.

Accept a JobSpec, persist it, execute it through the named runner with retries,
write every state transition to the store, and optionally schedule recurring
jobs. Runners stay dumb (signal in, result out); the orchestrator is where
retries, status, and durability live. In M2 a policy layer slots in front of
execution to enforce cross-job guardrails — which only composes because that
logic is centralized here, not scattered across runners."""

from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone

from apscheduler.schedulers.background import BackgroundScheduler

from .models import JobSpec, Run, RunResult, RunStatus
from .runner import RunnerRegistry
from .store import RunStore

log = logging.getLogger("mlcp.orchestrator")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Orchestrator:
    def __init__(
        self,
        store: RunStore,
        registry: RunnerRegistry,
        *,
        sync: bool = False,
        max_workers: int = 4,
    ) -> None:
        self.store = store
        self.registry = registry
        self.sync = sync
        self._pool = ThreadPoolExecutor(max_workers=max_workers)
        self._scheduler = BackgroundScheduler()
        self._futures: dict[str, Future] = {}

    # -- submission -------------------------------------------------------
    def submit(self, job: JobSpec) -> Run:
        """Accept a job and run it. Validates the runner up front so bad
        submissions fail fast. In async mode (server default) execution runs on
        the thread pool and this returns the PENDING run immediately; in sync
        mode (tests, CLI) it runs inline and returns the finished run."""
        self.registry.get(job.runner)  # raises KeyError before we persist anything
        run = Run(job=job)
        self.store.create_run(run)
        if self.sync:
            self._execute(run.id)
            refreshed = self.store.get_run(run.id)
            assert refreshed is not None  # we just created it
            return refreshed
        fut = self._pool.submit(self._execute, run.id)
        self._futures[run.id] = fut
        # Drop the future once it completes so _futures doesn't grow unbounded
        # under a long-running server or a recurring scheduled job.
        fut.add_done_callback(lambda _f, rid=run.id: self._futures.pop(rid, None))
        return run

    def wait(self, run_id: str, timeout: float | None = None) -> Run | None:
        """Block until a previously submitted async run completes (test/CLI helper)."""
        fut = self._futures.get(run_id)
        if fut is not None:
            fut.result(timeout=timeout)
        return self.store.get_run(run_id)

    # -- execution --------------------------------------------------------
    def _execute(self, run_id: str) -> None:
        run = self.store.get_run(run_id)
        if run is None:
            log.error("run %s disappeared before execution", run_id)
            return
        runner = self.registry.get(run.job.runner)
        run.started_at = _utcnow()
        max_attempts = run.job.max_retries + 1

        for attempt in range(1, max_attempts + 1):
            run.attempts = attempt
            run.status = RunStatus.RUNNING if attempt == 1 else RunStatus.RETRYING
            self.store.update_run(run)
            try:
                result = runner.run(run.job.params)
                run.result = result
                run.status = RunStatus.SUCCEEDED if result.ok else RunStatus.FAILED
            except Exception as exc:  # noqa: BLE001 - boundary: capture any runner error
                log.warning("run %s attempt %d raised: %s", run_id, attempt, exc)
                run.result = RunResult(ok=False, error=f"{type(exc).__name__}: {exc}")
                run.status = RunStatus.FAILED

            if run.status is RunStatus.SUCCEEDED:
                break
            if attempt < max_attempts:
                continue  # retry
            break  # out of attempts; stays FAILED

        run.finished_at = _utcnow()
        self.store.update_run(run)

    # -- scheduling -------------------------------------------------------
    def schedule(self, job: JobSpec) -> None:
        """Register a recurring job. Requires job.interval_seconds."""
        if not job.interval_seconds:
            raise ValueError("schedule() requires job.interval_seconds to be set.")
        self._scheduler.add_job(
            lambda: self.submit(job),
            trigger="interval",
            seconds=job.interval_seconds,
            id=f"job:{job.name}",
            replace_existing=True,
        )

    def start(self) -> None:
        if not self._scheduler.running:
            self._scheduler.start()

    def shutdown(self) -> None:
        if self._scheduler.running:
            self._scheduler.shutdown(wait=False)
        self._pool.shutdown(wait=False)
