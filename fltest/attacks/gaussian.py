"""Additive-Gaussian (Byzantine) model-poisoning attack.

Adds zero-mean Gaussian noise to the attacker's update, ``u' = u + N(0, sigma^2)``. One
of the "naive" attacks the proposal notes are over-used (Pitfall-1); included so the
testbed can reproduce and compare against stronger attacks.
"""

from __future__ import annotations

import numpy as np

from fltest.attacks.base import ThreatModelBaseClass
from fltest.core.hook_context import HookContext
from fltest.core.registry import register_attack


@register_attack("gaussian")
class GaussianAttack(ThreatModelBaseClass):
    HOOKS = ("after_client_train",)

    def __init__(self, sigma: float = 0.1, seed: int = 0, **params):
        super().__init__(**params)
        self.sigma = sigma
        self._rng = np.random.default_rng(seed)

    def after_client_train(self, ctx: HookContext) -> None:
        if not self.targets(ctx.client_id) or ctx.client_update is None:
            return
        ctx.client_update = [
            u + self._rng.normal(0.0, self.sigma, size=u.shape).astype(u.dtype)
            for u in ctx.client_update
        ]
