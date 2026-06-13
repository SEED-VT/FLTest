"""Gradient-perturbation defense (DP-style: clip then add Gaussian noise).

Clips the client's update *delta* (relative to the current global model) to a max L2 norm,
then adds zero-mean Gaussian noise — the user-space analogue of DP-SGD's per-update
clipping and noise. This is the slide-26 defense that blunts gradient-inversion (DLG):
with noise applied, DLG reconstruction MSE rises sharply.

Acts at ``after_client_train`` on the canonical list-of-ndarrays update.
"""

from __future__ import annotations

import numpy as np

from fltest.core.hook_context import HookContext
from fltest.core.registry import register_defense
from fltest.defenses.base import PPFLBaseClass


def _global_l2(arrays) -> float:
    return float(np.sqrt(sum(float(np.sum(a.astype(np.float64) ** 2)) for a in arrays)))


@register_defense("gradient_noise")
class GradientNoiseDefense(PPFLBaseClass):
    HOOKS = ("after_client_train",)

    def __init__(self, clip_norm: float = 1.0, sigma: float = 0.01, seed: int = 0, **params):
        super().__init__(**params)
        self.clip_norm = clip_norm
        self.sigma = sigma
        self._rng = np.random.default_rng(seed)

    def after_client_train(self, ctx: HookContext) -> None:
        if ctx.client_update is None:
            return
        update = ctx.client_update
        g = ctx.global_state
        # Work on the delta when we know the global model, else on the raw update.
        if g is not None and len(g) == len(update):
            delta = [u - gi for u, gi in zip(update, g)]
        else:
            delta, g = list(update), None

        norm = _global_l2(delta)
        scale = min(1.0, self.clip_norm / (norm + 1e-12))
        noised = [
            d * scale + self._rng.normal(0.0, self.sigma, size=d.shape).astype(d.dtype)
            for d in delta
        ]
        if g is not None:
            ctx.client_update = [gi + nd for gi, nd in zip(g, noised)]
        else:
            ctx.client_update = noised
