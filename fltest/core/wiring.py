"""Assemble a :class:`HookRunner` for a run from its :class:`RunSpec`.

Given a spec, this instantiates the metric listeners, attacks, and defenses named in the
config (looked up in the registries) and attaches their hooks, plus any user hook files
named in ``FLTEST_HOOKS``. Because it works purely from the (picklable) spec, the Flower
backend can call it again inside Ray workers to reconstruct client-side hooks — no need
to pickle handler closures across processes.

Registration order matters: attacks attach before defenses, so on a shared hook (e.g.
``after_client_train``) the attack tampers first and the defense sanitizes afterwards —
the realistic order.
"""

from __future__ import annotations

from fltest.core.config import RunSpec
from fltest.core.hook_runner import HookRunner


def build_hook_runner(spec: RunSpec, load_env_hooks: bool = True) -> HookRunner:
    """Build a HookRunner with all metrics/attacks/defenses/env-hooks attached."""
    # Importing these packages populates the registries (idempotent).
    import fltest.metrics  # noqa: F401
    import fltest.attacks  # noqa: F401
    import fltest.defenses  # noqa: F401
    from fltest.core.registry import ATTACKS, DEFENSES, METRICS

    runner = HookRunner()

    for metric_name in spec.metrics:
        METRICS.get(metric_name)().attach(runner)

    for atk in spec.attacks:
        ATTACKS.get(atk.name)(target_clients=atk.target_clients, **atk.params).attach(runner)

    for dfn in spec.defenses:
        DEFENSES.get(dfn.name)(**dfn.params).attach(runner)

    if load_env_hooks:
        from fltest.core import hooks

        hooks.import_convention_hooks()
        hooks.apply_to(runner)

    return runner
