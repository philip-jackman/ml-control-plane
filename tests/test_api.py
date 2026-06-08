"""Tests for the FastAPI control plane surface over the orchestrator."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from mlcp.api import create_app
from mlcp.orchestrator import Orchestrator
from mlcp.runner import RunnerRegistry
from mlcp.runners.builtin import FailingRunner, LinearFitRunner
from mlcp.store import RunStore


@pytest.fixture
def client(tmp_path) -> TestClient:
    store = RunStore(tmp_path / "test.db")
    registry = RunnerRegistry()
    registry.register(LinearFitRunner())
    registry.register(FailingRunner())
    # sync=True so POST /runs completes execution before responding.
    orchestrator = Orchestrator(store, registry, sync=True)
    app = create_app(orchestrator)
    return TestClient(app)


def _linear_fit_job() -> dict:
    return {
        "name": "fit",
        "runner": "linear_fit",
        "params": {"x": [1, 2, 3, 4], "y": [2, 4, 6, 8]},
    }


def test_health_ok(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_list_runners_includes_linear_fit(client: TestClient) -> None:
    resp = client.get("/runners")
    assert resp.status_code == 200
    assert "linear_fit" in resp.json()["runners"]


def test_post_run_succeeds_synchronously(client: TestClient) -> None:
    resp = client.post("/runs", json=_linear_fit_job())
    assert resp.status_code == 201
    body = resp.json()
    assert body["status"] == "succeeded"
    assert body["id"]


def test_list_runs_contains_submitted_run(client: TestClient) -> None:
    run_id = client.post("/runs", json=_linear_fit_job()).json()["id"]

    resp = client.get("/runs")
    assert resp.status_code == 200
    assert run_id in [r["id"] for r in resp.json()]


def test_get_run_by_id(client: TestClient) -> None:
    run_id = client.post("/runs", json=_linear_fit_job()).json()["id"]

    resp = client.get(f"/runs/{run_id}")
    assert resp.status_code == 200
    assert resp.json()["id"] == run_id


def test_get_run_missing_returns_404(client: TestClient) -> None:
    resp = client.get("/runs/nope")
    assert resp.status_code == 404


def test_post_run_unknown_runner_returns_400(client: TestClient) -> None:
    resp = client.post("/runs", json={"name": "ghost", "runner": "nope"})
    assert resp.status_code == 400
