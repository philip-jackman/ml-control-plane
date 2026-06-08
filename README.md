# ml-control-plane

A small, self-hostable control plane for running ML jobs through pluggable
runners under a centralized lifecycle. Submit a job and the orchestrator owns
everything that happens to it from there: status transitions, retries,
persistence, and the audit record. Runners stay dumb — signal in, result out —
and a FastAPI control plane sits on top as the HTTP surface. This is the public,
generic version of an orchestration pattern I run in regulated production, where
the same separation of concerns matters more, not less: runners produce results,
the orchestrator owns the run lifecycle, and a policy layer (M2) will own the
guardrails. What's working today is M1 — the runnable spine: you can install it,
run the tests, start the server, submit a real (if tiny) ML job, and read back
the full run record. No queue, no policy enforcement, no dashboard yet; those are
on the roadmap and the architecture is shaped to absorb them.

## Architecture

```
                          HTTP clients
                               │
                               ▼
              ┌─────────────────────────-───────┐
              │   FastAPI control plane         │
              │   /runs  /runners  /health      │  thin: validates + routes,
              │   (mlcp/api.py)                 │  holds no business logic
              └────────────────┬───────────-────┘
                               │
   ┌──────────────┐            ▼
   │ APScheduler  │   ┌───────────────────────-───────────┐
   │ (recurring   │──▶│  Orchestrator                     │
   │  jobs, by    │   │  owns the run lifecycle:          │
   │  interval)   │   │  status · retries · scheduling    │
   └──────────────┘   └───────┬──────────────-─────┬──────┘
                              │                   │
                              ▼                   ▼
                   ┌──────────────────┐  ┌─────────────-─────────┐
                   │ RunnerRegistry   │  │ RunStore (SQLite)     │
                   │ name -> Runner   │  │ the Run record IS the │
                   └────────┬─────────┘  │ audit trail (M1)      │
                            │            └──────────-────────────┘
                            ▼
        ┌─────────────────────────────────--──────────────┐
        │ Runners (dumb: params in, RunResult out)        │
        │  · linear_fit        — dependency-free OLS fit  │
        │  · always_fail       — exercises retry/failure  │
        │  · sklearn_classifier — optional [sklearn] extra│
        └─────────────────────────────────--──────────────┘
```

Every state transition a run goes through is written back to the `RunStore`, so
the persisted `Run` record — its status, attempt count, timestamps, and result —
is the audit trail in M1. Runners never touch status, retries, or persistence;
that is the orchestrator's job alone.

## Quickstart

Requires Python 3.11+.

```bash
# create and activate a virtualenv
python -m venv .venv
source .venv/bin/activate

# install in editable mode with dev tools
pip install -e ".[dev]"

# optional: add the scikit-learn runner extra
pip install -e ".[sklearn]"

# run the tests
pytest

# start the server (either form works)
mlcp
# or
python -m mlcp
```

The server binds to `127.0.0.1:8000` by default. Configure it with environment
variables:

| Variable    | Default       | Purpose                          |
|-------------|---------------|----------------------------------|
| `MLCP_DB`   | `mlcp.db`     | Path to the SQLite database file |
| `MLCP_HOST` | `127.0.0.1`   | Bind host                        |
| `MLCP_PORT` | `8000`        | Bind port                        |

With the server running, submit a job and read it back:

```bash
# submit a linear_fit job; returns 201 with the PENDING run (id included)
curl -s -X POST http://127.0.0.1:8000/runs \
  -H 'content-type: application/json' \
  -d '{
        "name": "demo-line-fit",
        "runner": "linear_fit",
        "params": {"x": [1, 2, 3, 4], "y": [2, 4, 6, 8]},
        "max_retries": 1
      }'

# list runs (newest first; optional ?status= and ?limit= filters)
curl -s http://127.0.0.1:8000/runs

# fetch a single run by id — the full audit record, including result + metrics
curl -s http://127.0.0.1:8000/runs/<run_id>
```

`GET /runners` lists the registered runners (the `sklearn_classifier` runner only
appears if the `[sklearn]` extra is installed). `GET /health` is a liveness probe.

## Design decisions worth defending

- **The run lifecycle is centralized, not per-runner.** Status transitions,
  retries, and timestamps live in exactly one place — the orchestrator. Runners
  declare a `name` and implement `run`; they return a `RunResult` or raise. That
  is the entire contract. The payoff is that the M2 policy layer (concurrency
  caps, quotas, approval gates) slots in front of one execution path instead of
  being reimplemented inside every runner.

- **Runners are pluggable and isolate their own dependencies.** Adding a runner
  is subclassing `Runner` and registering it. Heavy runners import their deps
  lazily and ship as optional extras, so a missing extra (`scikit-learn`) is a
  disabled runner, not a broken install. The core platform runs and tests with
  nothing but FastAPI, Pydantic, and APScheduler.

- **The persisted `Run` record is the M1 audit trail.** Every transition is
  written through the store, so the record on disk is the source of truth for
  what happened: which runner, how many attempts, when it started and finished,
  and the result or error. There is no separate "audit mode" to forget to turn
  on.

- **SQLite now, but a Postgres-shaped store API.** The `RunStore` surface is four
  methods (`create_run`, `update_run`, `get_run`, `list_runs`). The full run is a
  JSON document with `status` and `created_at` denormalized into indexed columns
  for cheap filtering and ordering. Swapping the backend later means
  reimplementing those four methods, not rewriting callers.

## What I'd build differently at scale

This is a single-process control plane on purpose — it has to be small enough to
read in one sitting. At real volume I'd change:

- **Process-per-runner services behind a queue.** Today execution runs on a
  thread pool in-process. At scale, runners become independent workers pulling
  from a durable queue, so a heavy or wedged runner can't starve the control
  plane and capacity scales per runner type.
- **Run records become a structured event stream with real observability.** The
  single mutable `Run` row is fine for M1, but a transition log (emitted as
  structured events) plus metrics and tracing is what you actually operate
  against in production.
- **A model registry.** Versioned models with lineage from training run to
  deployed artifact, so a runner references a model by id and version rather than
  loading whatever is on disk.
- **Drift detection.** Monitor input distributions and output quality over time
  and feed that back into the policy layer to gate or alert on degradation.

## Roadmap

- **M1 — runnable spine (done).** Orchestrator-owned lifecycle, pluggable
  runners, SQLite-backed run records as the audit trail, FastAPI control plane,
  interval scheduling via APScheduler.
- **M2 — centralized policy / guardrail layer.** Concurrency caps, per-job
  quotas, allow/deny rules, and approval gates enforced in front of execution,
  plus structured audit events replacing the single mutable run row.
- **M3 — operability.** Output and health monitoring, a dashboard over runs and
  runners, and dev -> prod config promotion.

## License

MIT. See [LICENSE](LICENSE).
