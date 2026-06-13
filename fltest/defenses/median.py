"""Coordinate-wise median robust aggregation (Yin et al., ICML 2018).

Aggregates by taking, for each parameter coordinate, the median across client updates.
Simple and robust to a Byzantine minority.
"""

from __future__ import annotations

import numpy as np

from fltest.core.hook_context import HookContext
from fltest.core.registry import register_defense
from fltest.defenses._robust import replace_with, stack, unflatten
from fltest.defenses.base import PPFLBaseClass


@register_defense("median")
class MedianDefense(PPFLBaseClass):
    HOOKS = ("before_aggregate",)

    def before_aggregate(self, ctx: HookContext) -> None:
        uw = ctx.updates_and_weights
        if not uw:
            return
        updates = [u for u, _ in uw]
        agg = np.median(stack(updates), axis=0)
        replace_with(ctx, unflatten(agg, updates[0]))
