"""ml-control-plane: a small, self-hostable control plane for running ML jobs
through pluggable runners under a centralized lifecycle and audit trail.

The public, generic version of the orchestration pattern I run in production:
runners produce results; the orchestrator owns the lifecycle (retries, status,
persistence); a future policy layer owns the guardrails. Separation of concerns
is the whole point.
"""

__version__ = "0.1.0"
