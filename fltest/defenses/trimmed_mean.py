"""Coordinate-wise trimmed-mean robust aggregation (Yin et al., ICML 2018).

For each parameter coordinate, drops the ``trim`` largest and ``trim`` smallest values
across clients, then averages the rest — discarding the extreme values a Byzantine client
would inject.
"""

from __future__ import annotations

import numpy as np

from fltest.core.hook_context import HookContext
from fltest.core.registry import register_defense
from fltest.defenses._robust import replace_with, stack, unflatten
from fltest.defenses.base import PPFLBaseClass


@register_defense("trimmed_mean")
class TrimmedMeanDefense(PPFLBaseClass):
    HOOKS = ("before_aggregate",)

    def __init__(self, trim: int = 1, **params):
        super().__init__(**params)
        self.trim = trim

    def before_aggregate(self, ctx: HookContext) -> None:
        uw = ctx.updates_and_weights
        if not uw:
            return
        updates = [u for u, _ in uw]
        mat = stack(updates)  # (n, dim)
        n = mat.shape[0]
        k = min(self.trim, (n - 1) // 2)
        smat = np.sort(mat, axis=0)
        kept = smat[k : n - k] if k > 0 else smat
        agg = kept.mean(axis=0)
        replace_with(ctx, unflatten(agg, updates[0]))
