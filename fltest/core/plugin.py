"""Shared base for hook-based plugins (attacks, defenses, metric listeners).

A plugin declares which lifecycle hooks it implements in ``HOOKS`` and provides matching
methods; :meth:`attach` registers exactly those onto a :class:`HookRunner`. This single
mechanism backs all three extension points so they compose uniformly on one run.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from fltest.core.hook_context import HookContext
from fltest.core.hook_runner import HookRunner

LIFECYCLE_HOOKS: Tuple[str, ...] = (
    "before_simulation",
    "on_data_distribute",
    "before_round",
    "before_client_train",
    "after_client_train",
    "before_aggregate",
    "on_aggregate",
    "after_aggregate",
    "after_round",
    "after_simulation",
)


class HookPlugin:
    """Base for any object that contributes lifecycle hooks to a run."""

    #: Subset of LIFECYCLE_HOOKS this plugin implements.
    HOOKS: Tuple[str, ...] = ()

    def __init__(self, target_clients=None, **params: Any):
        self.target_clients = set(target_clients) if target_clients else None
        self.params: Dict[str, Any] = params

    def targets(self, client_id) -> bool:
        """True if this plugin should act on ``client_id`` (None target => all clients)."""
        return self.target_clients is None or client_id in self.target_clients

    def attach(self, runner: HookRunner) -> None:
        for name in self.HOOKS:
            if name not in LIFECYCLE_HOOKS:
                raise ValueError(f"{type(self).__name__}: unknown hook '{name}'")
            method = getattr(self, name, None)
            if not callable(method):
                raise ValueError(
                    f"{type(self).__name__}: HOOKS lists '{name}' but no method defined"
                )
            runner.register(name, method)

    # Default no-op lifecycle methods; subclasses override only what they declare in HOOKS.
    def before_simulation(self, ctx: HookContext) -> None: ...
    def on_data_distribute(self, ctx: HookContext) -> None: ...
    def before_round(self, ctx: HookContext) -> None: ...
    def before_client_train(self, ctx: HookContext) -> None: ...
    def after_client_train(self, ctx: HookContext) -> None: ...
    def before_aggregate(self, ctx: HookContext) -> None: ...
    def on_aggregate(self, ctx: HookContext) -> None: ...
    def after_aggregate(self, ctx: HookContext) -> None: ...
    def after_round(self, ctx: HookContext) -> None: ...
    def after_simulation(self, ctx: HookContext) -> None: ...
