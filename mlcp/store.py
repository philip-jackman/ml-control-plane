"""Persistence for Run records.

The full Run is stored as a JSON document; `status` and `created_at` are
denormalized into columns so filtering and ordering stay cheap. A new SQLite
connection is opened per operation, which keeps the store safe to call from the
orchestrator's worker threads without shared-connection footguns. The public
API is deliberately small and Postgres-shaped — swapping the backend later means
reimplementing four methods, not rewriting callers."""

from __future__ import annotations

import sqlite3
from datetime import timezone
from pathlib import Path
from typing import Optional

from .models import Run, RunStatus

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id          TEXT PRIMARY KEY,
    status      TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    document    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);
CREATE INDEX IF NOT EXISTS idx_runs_created_at ON runs(created_at);
"""


class RunStore:
    def __init__(self, db_path: str | Path = "mlcp.db") -> None:
        self.db_path = str(db_path)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def create_run(self, run: Run) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO runs (id, status, created_at, document) VALUES (?, ?, ?, ?)",
                (
                    run.id,
                    run.status.value,
                    # Normalize to UTC so the indexed column sorts chronologically
                    # even if a naive or non-UTC datetime ever reaches the boundary.
                    run.created_at.astimezone(timezone.utc).isoformat(),
                    run.model_dump_json(),
                ),
            )

    def update_run(self, run: Run) -> None:
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE runs SET status = ?, document = ? WHERE id = ?",
                (run.status.value, run.model_dump_json(), run.id),
            )
            if cur.rowcount == 0:
                raise KeyError(f"Run '{run.id}' not found.")

    def get_run(self, run_id: str) -> Optional[Run]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT document FROM runs WHERE id = ?", (run_id,)
            ).fetchone()
        return Run.model_validate_json(row["document"]) if row else None

    def list_runs(
        self, status: Optional[RunStatus] = None, limit: int = 100
    ) -> list[Run]:
        query = "SELECT document FROM runs"
        args: list[object] = []
        if status is not None:
            query += " WHERE status = ?"
            args.append(status.value)
        query += " ORDER BY created_at DESC, id DESC LIMIT ?"
        args.append(limit)
        with self._connect() as conn:
            rows = conn.execute(query, args).fetchall()
        return [Run.model_validate_json(r["document"]) for r in rows]
