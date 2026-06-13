"""Shared helpers for robust-aggregation defenses (Krum, trimmed-mean, median).

These operate at ``before_aggregate`` by computing a single robust aggregate from all
client updates and replacing ``ctx.updates_and_weights`` with ``[(robust, total_n)]`` so
the backend's subsequent average returns exactly the robust result.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def flatten(update: List[np.ndarray]) -> np.ndarray:
    return np.concatenate([a.astype(np.float64).ravel() for a in update])


def unflatten(vec: np.ndarray, template: List[np.ndarray]) -> List[np.ndarray]:
    out, i = [], 0
    for a in template:
        n = a.size
        out.append(vec[i : i + n].reshape(a.shape).astype(a.dtype))
        i += n
    return out


def stack(updates: List[List[np.ndarray]]) -> np.ndarray:
    return np.stack([flatten(u) for u in updates], axis=0)  # (num_clients, dim)


def replace_with(ctx, robust_update: List[np.ndarray]) -> None:
    total = sum(n for _, n in ctx.updates_and_weights)
    ctx.updates_and_weights = [(robust_update, total)]
