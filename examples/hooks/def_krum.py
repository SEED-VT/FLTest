"""Krum / Multi-Krum Byzantine-robust aggregation defense.

Reference:
  Blanchard, El Mhamdi, Guerraoui, Stainer -- "Machine Learning with
  Adversaries: Byzantine Tolerant Gradient Descent" (NeurIPS 2017).
  https://papers.nips.cc/paper/2017/hash/f4b3edd58351508562bf404775a97f11-Paper.pdf

Algorithm. Given n client updates and an assumed Byzantine count f:
  score(i) = sum of squared L2 distances from update i to its
             (n - f - 2) closest neighbors.
Classic Krum returns the single update with the lowest score.
Multi-Krum averages the m lowest-score updates (m = 1 is classic Krum,
m > 1 trades some Byzantine robustness for variance reduction).

We operate on full client models (parameters), not deltas. This is
mathematically equivalent because all clients in a given round start
from the same global state, so pairwise L2 distances between models
equal pairwise L2 distances between deltas.

Guard rails per the paper:
  n >= 2 * f + 3
  m <= n - f - 2

Load via:
  export FLTEST_HOOKS=examples/hooks/def_krum
  poetry run python fltest/main.py

Combine with an attack hook by comma-separating:
  export FLTEST_HOOKS=examples/hooks/atk_backdoor,examples/hooks/def_krum
"""

import numpy as np

from fltest.core import hooks


# Configuration
KRUM_F = 1   # assumed maximum number of Byzantine clients
KRUM_M = 1   # 1 = classic Krum; >= 2 = Multi-Krum


def _flatten(update):
    """Pack a list of ndarrays into one float64 1-D vector + (shapes, dtypes)."""
    shapes = [a.shape for a in update]
    dtypes = [a.dtype for a in update]
    flat = np.concatenate([a.ravel().astype(np.float64) for a in update])
    return flat, shapes, dtypes


def _unflatten(flat, shapes, dtypes):
    out = []
    offset = 0
    for shape, dtype in zip(shapes, dtypes):
        size = int(np.prod(shape)) if shape else 1
        chunk = flat[offset:offset + size].reshape(shape)
        out.append(chunk.astype(dtype))
        offset += size
    return out


def _krum_scores(flats, f):
    """For each i, sum of squared L2 distances to (n - f - 2) closest others."""
    n = len(flats)
    closest_k = n - f - 2
    sq_dists = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.sum((flats[i] - flats[j]) ** 2))
            sq_dists[i, j] = d
            sq_dists[j, i] = d
    scores = np.empty(n, dtype=np.float64)
    for i in range(n):
        others = np.delete(sq_dists[i], i)
        others.sort()
        scores[i] = others[:closest_k].sum()
    return scores


@hooks.on_aggregate
def krum_aggregate(ctx):
    if not ctx.updates_and_weights:
        return

    updates = [u for u, _ in ctx.updates_and_weights]
    n = len(updates)
    f = KRUM_F
    m = KRUM_M

    if n < 2 * f + 3:
        raise ValueError(
            f"[KRUM] requires n >= 2*f + 3 (got n={n}, f={f}). "
            "Reduce KRUM_F or increase the number of participating clients."
        )
    if m < 1 or m > n - f - 2:
        raise ValueError(
            f"[KRUM] requires 1 <= m <= n - f - 2 (got m={m}, n={n}, f={f}, "
            f"max m={n - f - 2}). Adjust KRUM_M."
        )

    flat_shapes_dtypes = [_flatten(u) for u in updates]
    flats = [fsd[0] for fsd in flat_shapes_dtypes]
    shapes = flat_shapes_dtypes[0][1]
    dtypes = flat_shapes_dtypes[0][2]

    scores = _krum_scores(flats, f)
    top_m_idx = np.argsort(scores)[:m]
    avg_flat = np.mean([flats[i] for i in top_m_idx], axis=0)

    ctx.new_global_state = _unflatten(avg_flat, shapes, dtypes)

    variant = "Krum" if m == 1 else f"Multi-Krum (m={m})"
    print(f"  [KRUM] {variant} round={ctx.round} n={n} f={f} "
          f"selected={top_m_idx.tolist()} scores_min={scores.min():.4g} "
          f"scores_max={scores.max():.4g}")
