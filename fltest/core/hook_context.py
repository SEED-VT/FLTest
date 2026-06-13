"""Mutable context passed to every hook.

A single ``HookContext`` instance flows through the FL lifecycle. Each lifecycle phase
populates the fields relevant to it; hooks (attacks, defenses, validators, metric
listeners) read and mutate those fields. Because every backend emits the same hooks with
the same context shape, a hook written once runs unchanged across all frameworks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


@dataclass
class HookContext:
    """Context for hook handlers. Fields are optional; only set what each phase needs."""

    # --- run identity / config ---
    cfg: Optional[Any] = None            # the resolved RunSpec for this run
    framework: Optional[str] = None      # backend name, e.g. "reference", "flwr"
    run_name: Optional[str] = None       # human label for the run
    round: Optional[int] = None
    client_id: Optional[int] = None

    # --- data phase ---
    raw_dataset: Optional[Any] = None
    partition_map: Optional[Dict[int, Any]] = None
    dist_dict: Optional[Dict[int, Any]] = None   # cid -> client train data/loader
    test_data: Optional[Any] = None

    # --- model / update state ---
    client_data: Optional[Any] = None            # this client's train loader (mutable by attacks)
    global_state: Optional[Any] = None           # current global params (list[ndarray] or state_dict)
    client_update: Optional[Any] = None           # this client's update (mutable by attacks/defenses)
    num_samples: Optional[int] = None
    updates_and_weights: Optional[List[tuple]] = None  # [(update, n), ...] at aggregation
    new_global_state: Optional[Any] = None
    model: Optional[Any] = None                   # live nn.Module when a backend can expose it

    # --- attack/defense scratch space (gradient inversion etc.) ---
    true_grads: Optional[Any] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    # --- metrics / outputs ---
    metrics: Dict[str, Any] = field(default_factory=dict)        # current-phase metrics
    history: Dict[int, Dict[str, Any]] = field(default_factory=dict)  # round -> metrics
    listeners: List[Callable[["HookContext"], None]] = field(default_factory=list)

    # Final accuracy convenience field (set at after_simulation); used by diff/metamorphic hooks.
    final_accuracy: Optional[float] = None
    all_frameworks: Optional[List[str]] = None

    def record(self, **kv: Any) -> None:
        """Merge key/values into ``metrics`` and into ``history[round]`` when a round is set."""
        self.metrics.update(kv)
        if self.round is not None:
            self.history.setdefault(self.round, {}).update(kv)
