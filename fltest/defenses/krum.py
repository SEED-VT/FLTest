"""Krum robust-aggregation defense (Blanchard et al., NeurIPS 2017).

Selects the single client update whose squared distance to its ``n - f - 2`` nearest
neighbours is smallest — the update most "agreed upon" by the honest majority — and uses
it as the round's aggregate. Robust to up to ``f`` Byzantine clients.
"""

from __future__ import annotations

import numpy as np

from fltest.core.hook_context import HookContext
from fltest.core.registry import register_defense
from fltest.defenses._robust import replace_with, stack
from fltest.defenses.base import PPFLBaseClass


@register_defense("krum")
class KrumDefense(PPFLBaseClass):
    HOOKS = ("before_aggregate",)

    def __init__(self, num_byzantine: int = 1, **params):
        super().__init__(**params)
        self.f = num_byzantine

    def before_aggregate(self, ctx: HookContext) -> None:
        uw = ctx.updates_and_weights
        if not uw or len(uw) < 3:
            return
        updates = [u for u, _ in uw]
        mat = stack(updates)  # (n, dim)
        n = mat.shape[0]
        # squared pairwise distances
        sq = np.sum((mat[:, None, :] - mat[None, :, :]) ** 2, axis=-1)
        m = max(1, n - self.f - 2)
        scores = []
        for i in range(n):
            dists = np.sort(sq[i])[1 : m + 1]  # exclude self (distance 0)
            scores.append(dists.sum())
        chosen = int(np.argmin(scores))
        replace_with(ctx, updates[chosen])
