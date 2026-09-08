"""Model-replacement attack for surviving federated averaging.

After training a malicious local model, boost its delta from the current global model:

``submitted = global + scale * (local - global)``

With equal client weights and negligible benign deltas, one attacker uses a scale equal to
the number of participating clients to make the aggregate land on its local model. Multiple
colluding attackers divide that scale among themselves. This is the train-and-scale model
replacement attack of Bagdasaryan et al. (AISTATS 2020).

FLTest's client hook does not expose the total sample weight selected for the current round,
so automatic scaling is exact only for full-participation, equal-weight aggregation. Set
``scale`` explicitly for sample-weighted or otherwise customized aggregation.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from fltest.attacks.base import ThreatModelBaseClass
from fltest.core.hook_context import HookContext
from fltest.core.registry import register_attack


@register_attack("model_replacement")
class ModelReplacementAttack(ThreatModelBaseClass):
    """Boost malicious local-model deltas before they are submitted for aggregation."""

    HOOKS = ("after_client_train",)

    def __init__(
        self,
        scale: Optional[float] = None,
        target_round: Optional[int] = None,
        **params,
    ):
        super().__init__(**params)
        if scale is not None and scale <= 0:
            raise ValueError("model_replacement scale must be greater than zero")
        if target_round is not None and target_round < 1:
            raise ValueError("model_replacement target_round must be at least 1")
        self.scale = scale
        self.target_round = target_round

    def _effective_scale(self, ctx: HookContext) -> float:
        if self.scale is not None:
            return float(self.scale)

        num_clients = getattr(ctx.cfg, "num_clients", None)
        if not isinstance(num_clients, int) or num_clients < 1:
            raise ValueError(
                "model_replacement needs scale or cfg.num_clients for automatic scaling"
            )
        num_attackers = len(self.target_clients) if self.target_clients is not None else num_clients
        return float(num_clients / num_attackers)

    def after_client_train(self, ctx: HookContext) -> None:
        if not self.targets(ctx.client_id):
            return
        if self.target_round is not None and ctx.round != self.target_round:
            return
        if ctx.client_update is None or ctx.global_state is None:
            return
        if len(ctx.client_update) != len(ctx.global_state):
            raise ValueError("model_replacement requires matching local and global states")

        scale = self._effective_scale(ctx)
        boosted = []
        for local, global_ in zip(ctx.client_update, ctx.global_state):
            local_arr = np.asarray(local)
            global_arr = np.asarray(global_)
            if local_arr.shape != global_arr.shape:
                raise ValueError("model_replacement requires matching parameter shapes")
            replaced = global_arr + scale * (local_arr - global_arr)
            boosted.append(replaced.astype(local_arr.dtype, copy=False))

        ctx.client_update = boosted
        ctx.record(model_replacement_scale=scale)
