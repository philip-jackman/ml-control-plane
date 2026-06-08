"""The runner contract and registry.

A runner is the only thing a user writes to extend the platform. It declares a
unique `name` and implements `run`. It returns a RunResult on success or raises
on failure — it never touches retries, status, scheduling, or persistence. Those
are the orchestrator's job. This separation is what lets a future policy layer
enforce limits across all runners uniformly."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .models import RunResult


class Runner(ABC):
    """A unit of work the control plane knows how to execute."""

    #: Unique key the registry and JobSpec.runner reference. Subclasses must set it.
    name: str = ""

    @abstractmethod
    def run(self, params: dict[str, Any]) -> RunResult:
        """Execute the job. Return a RunResult, or raise to signal failure."""
        raise NotImplementedError


class RunnerRegistry:
    """Maps runner name -> runner instance. The orchestrator resolves runners
    through this; nothing else holds a direct reference to a runner."""

    def __init__(self) -> None:
        self._runners: dict[str, Runner] = {}

    def register(self, runner: Runner) -> Runner:
        if not runner.name:
            raise ValueError(f"{type(runner).__name__} must set a non-empty `name`.")
        if runner.name in self._runners:
            raise ValueError(f"Runner '{runner.name}' is already registered.")
        self._runners[runner.name] = runner
        return runner

    def get(self, name: str) -> Runner:
        try:
            return self._runners[name]
        except KeyError:
            raise KeyError(
                f"No runner registered under '{name}'. Known runners: {self.names()}"
            ) from None

    def names(self) -> list[str]:
        return sorted(self._runners)
