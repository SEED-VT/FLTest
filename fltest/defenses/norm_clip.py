"""Norm-clipping defense.

Clips each client's update delta to a maximum L2 norm — a standard mitigation against
model-poisoning attacks that send large-magnitude updates (e.g. sign-flip, scaled
backdoors). Acts at ``after_client_train``; like a DP defense without the noise term.
"""

from __future__ import annotations

import numpy as np

from fltest.core.hook_context import HookContext
from fltest.core.registry import register_defense
from fltest.defenses.base import PPFLBaseClass
from fltest.defenses.gradient_noise import _global_l2


@register_defense("norm_clip")
class NormClipDefense(PPFLBaseClass):
    HOOKS = ("after_client_train",)

    def __init__(self, clip_norm: float = 1.0, **params):
        super().__init__(**params)
        self.clip_norm = clip_norm

    def after_client_train(self, ctx: HookContext) -> None:
        if ctx.client_update is None:
            return
        update = ctx.client_update
        g = ctx.global_state
        if g is not None and len(g) == len(update):
            delta = [u - gi for u, gi in zip(update, g)]
            scale = min(1.0, self.clip_norm / (_global_l2(delta) + 1e-12))
            ctx.client_update = [gi + d * scale for gi, d in zip(g, delta)]
        else:
            scale = min(1.0, self.clip_norm / (_global_l2(update) + 1e-12))
            ctx.client_update = [u * scale for u in update]
