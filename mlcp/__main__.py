"""Entry point: wire the store, registry, orchestrator, and API together and
serve. Configuration is via environment variables so the same image runs in any
environment (the dev -> prod promotion story lands properly in M3).

    MLCP_DB    path to the SQLite database (default: mlcp.db)
    MLCP_HOST  bind host (default: 127.0.0.1)
    MLCP_PORT  bind port (default: 8000)
"""

from __future__ import annotations

import logging
import os

import uvicorn

from .api import create_app
from .orchestrator import Orchestrator
from .runner import RunnerRegistry
from .runners.builtin import FailingRunner, LinearFitRunner
from .store import RunStore


def build_registry() -> RunnerRegistry:
    """Register the always-available runners, plus the sklearn runner only if
    its optional dependency is installed."""
    registry = RunnerRegistry()
    registry.register(LinearFitRunner())
    registry.register(FailingRunner())
    try:
        from .runners.sklearn_runner import SklearnClassifierRunner
    except Exception:  # noqa: BLE001 - optional extra not installed; skip it
        log = logging.getLogger("mlcp")
        log.info("scikit-learn extra not installed; sklearn_classifier runner disabled.")
    else:
        registry.register(SklearnClassifierRunner())
    return registry


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    db_path = os.environ.get("MLCP_DB", "mlcp.db")
    host = os.environ.get("MLCP_HOST", "127.0.0.1")
    port = int(os.environ.get("MLCP_PORT", "8000"))

    store = RunStore(db_path)
    orchestrator = Orchestrator(store, build_registry())
    orchestrator.start()
    app = create_app(orchestrator)
    try:
        uvicorn.run(app, host=host, port=port)
    finally:
        orchestrator.shutdown()


if __name__ == "__main__":
    main()
