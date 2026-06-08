"""FastAPI control plane: the HTTP surface over the orchestrator.

Thin by design — every route delegates to the orchestrator or the store. The
control plane validates and routes; it holds no business logic of its own."""

from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, HTTPException, Query

from .models import JobSpec, Run, RunStatus
from .orchestrator import Orchestrator


def create_app(orchestrator: Orchestrator) -> FastAPI:
    app = FastAPI(title="ml-control-plane", version="0.1.0")
    app.state.orchestrator = orchestrator

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/runners")
    def list_runners() -> dict[str, list[str]]:
        return {"runners": orchestrator.registry.names()}

    @app.post("/runs", response_model=Run, status_code=201)
    def submit_run(job: JobSpec) -> Run:
        try:
            return orchestrator.submit(job)
        except KeyError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/runs", response_model=list[Run])
    def list_runs(
        status: Optional[RunStatus] = None,
        limit: int = Query(default=100, ge=1, le=1000),
    ) -> list[Run]:
        return orchestrator.store.list_runs(status=status, limit=limit)

    @app.get("/runs/{run_id}", response_model=Run)
    def get_run(run_id: str) -> Run:
        run = orchestrator.store.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
        return run

    return app
