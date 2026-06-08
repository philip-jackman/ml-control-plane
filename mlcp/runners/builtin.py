"""Dependency-free example runners. These exist so the whole platform runs and
its tests pass with nothing but the core deps installed."""

from __future__ import annotations

from typing import Any

from ..models import RunResult
from ..runner import Runner


class LinearFitRunner(Runner):
    """Fit y = slope * x + intercept by ordinary least squares.

    A real (if tiny) model with no third-party dependencies — enough to show the
    platform executing an ML job, computing metrics, and persisting results end
    to end. Heavier models live in optional runners.
    """

    name = "linear_fit"

    def run(self, params: dict[str, Any]) -> RunResult:
        xs = [float(v) for v in params.get("x", [])]
        ys = [float(v) for v in params.get("y", [])]
        if len(xs) != len(ys):
            raise ValueError("'x' and 'y' must be the same length.")
        if len(xs) < 2:
            raise ValueError("need at least two points to fit a line.")

        n = len(xs)
        mean_x = sum(xs) / n
        mean_y = sum(ys) / n
        sxx = sum((x - mean_x) ** 2 for x in xs)
        if sxx == 0.0:
            raise ValueError("'x' has zero variance; cannot fit a line.")

        slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / sxx
        intercept = mean_y - slope * mean_x

        ss_tot = sum((y - mean_y) ** 2 for y in ys)
        ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
        r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0

        return RunResult(
            ok=True,
            outputs={"slope": slope, "intercept": intercept},
            metrics={"r2": r2, "n": float(n)},
        )


class FailingRunner(Runner):
    """Always raises. Exists to exercise the orchestrator's retry and failure
    paths in tests and demos."""

    name = "always_fail"

    def run(self, params: dict[str, Any]) -> RunResult:
        raise RuntimeError(params.get("message", "intentional failure"))
