"""The framework abstraction: every FL backend implements ``run_simulation``.

This is the ``run_simulation()`` boundary from the Q1 slides — the single seam between
FLTest's core (hooks, attacks, defenses, metrics, tests) and any concrete FL framework.
The core never imports Flower/NVFlare/pfl directly; it asks the registry for an adapter
by name and calls :meth:`FrameworkAdapter.run_simulation`.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from fltest.core import HookRunner
from fltest.core.config import RunSpec
from fltest.core.registry import FRAMEWORKS


@dataclass
class RunResult:
    """Outcome of a single FL run, framework-agnostic."""

    run_id: str
    run_name: str
    framework: str
    status: str = "success"  # success | failed
    error: Optional[str] = None
    duration_seconds: float = 0.0
    final: Dict[str, float] = field(default_factory=dict)         # e.g. {"accuracy":.., "loss":..}
    history: Dict[int, Dict[str, Any]] = field(default_factory=dict)  # round -> metrics
    per_client: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_name": self.run_name,
            "framework": self.framework,
            "status": self.status,
            "error": self.error,
            "duration_seconds": round(self.duration_seconds, 3),
            "final": self.final,
            "history": {str(k): v for k, v in self.history.items()},
            "per_client": {str(k): v for k, v in self.per_client.items()},
            "extras": self.extras,
        }


class FrameworkAdapter(ABC):
    """Base class for FL framework adapters.

    Subclasses run a full federated simulation for ``spec`` and must emit the lifecycle
    hooks (``before_simulation`` → ``after_simulation``) so attacks/defenses/metric
    listeners registered on ``hook_runner`` execute at the right points.
    """

    #: Backend name used in configs and the registry.
    name: str = "base"

    @abstractmethod
    def run_simulation(self, spec: RunSpec, data: Dict[str, Any], hook_runner: HookRunner) -> RunResult:
        """Execute the FL simulation. ``data`` is the prepared dataset bundle for ``spec``."""
        raise NotImplementedError

    # Convenience timing wrapper used by subclasses / orchestrator.
    @staticmethod
    def timed(fn, *args, **kwargs):
        start = time.time()
        out = fn(*args, **kwargs)
        return out, time.time() - start


def get_adapter(name: str) -> FrameworkAdapter:
    """Instantiate the registered adapter ``name`` (raises a clear error if missing)."""
    cls = FRAMEWORKS.get(name)
    return cls()
