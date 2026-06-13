"""Sign-flipping model-poisoning attack.

Reflects the client's update around the current global model and scales it, pushing
aggregation in the opposite direction: ``u' = g - scale * (u - g)``. If the global state
is unavailable, falls back to negating the update. Acts at ``after_client_train`` on the
canonical list-of-ndarrays update.
"""

from __future__ import annotations

import numpy as np

from fltest.attacks.base import ThreatModelBaseClass
from fltest.core.hook_context import HookContext
from fltest.core.registry import register_attack


@register_attack("sign_flip")
class SignFlipAttack(ThreatModelBaseClass):
    HOOKS = ("after_client_train",)

    def __init__(self, scale: float = 1.0, **params):
        super().__init__(**params)
        self.scale = scale

    def after_client_train(self, ctx: HookContext) -> None:
        if not self.targets(ctx.client_id) or ctx.client_update is None:
            return
        update = ctx.client_update
        g = ctx.global_state
        if g is not None and len(g) == len(update):
            ctx.client_update = [gi - self.scale * (ui - gi) for ui, gi in zip(update, g)]
        else:
            ctx.client_update = [-self.scale * ui for ui in update]
