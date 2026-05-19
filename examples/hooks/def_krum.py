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


def _validate_layout(updates):
    """All clients must have identical per-tensor (shape, dtype) layouts."""
    ref_shapes = [a.shape for a in updates[0]]
    ref_dtypes = [a.dtype for a in updates[0]]
    for k, upd in enumerate(updates[1:], start=1):
        if len(upd) != len(updates[0]):
            raise ValueError(
                f"[KRUM] client {k} has {len(upd)} tensors; "
                f"expected {len(updates[0])} (matching client 0)."
            )
        for j, (a, ref_s, ref_d) in enumerate(zip(upd, ref_shapes, ref_dtypes)):
            if a.shape != ref_s:
                raise ValueError(
                    f"[KRUM] client {k} tensor {j} has shape {a.shape}; "
                    f"expected {ref_s}."
                )
            if a.dtype != ref_d:
                raise ValueError(
                    f"[KRUM] client {k} tensor {j} has dtype {a.dtype}; "
                    f"expected {ref_d}."
                )


def _flatten_floats(update, float_idx):
    """Concatenate only the floating-point tensors into one float64 vector.

    Score computation needs a single distance vector, but Krum is defined on
    parameters and integer buffers (e.g. BatchNorm num_batches_tracked) carry
    no gradient signal -- treating them as parameters distorts distances and
    averaging them across selected clients would silently truncate. Excluding
    them from both the score and the aggregate keeps int buffers exact.
    """
    return np.concatenate([update[j].ravel().astype(np.float64) for j in float_idx])


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

    _validate_layout(updates)

    # Partition tensor indices by dtype: floats participate in score and
    # aggregation; ints (e.g. BN num_batches_tracked) pass through from the
    # first selected client.
    float_idx = [j for j, a in enumerate(updates[0])
                 if np.issubdtype(a.dtype, np.floating)]
    int_idx = [j for j in range(len(updates[0])) if j not in float_idx]

    flats = [_flatten_floats(u, float_idx) for u in updates]
    scores = _krum_scores(flats, f)
    top_m_idx = np.argsort(scores)[:m]

    # Aggregate per-tensor.
    out = [None] * len(updates[0])
    for j in float_idx:
        if m == 1:
            out[j] = updates[top_m_idx[0]][j].copy()
        else:
            avg = np.mean(
                [updates[i][j].astype(np.float64) for i in top_m_idx],
                axis=0,
            )
            out[j] = avg.astype(updates[0][j].dtype)
    for j in int_idx:
        # Pass through from the first selected client -- averaging integer
        # state would either truncate (losing info) or float-bleed.
        out[j] = updates[top_m_idx[0]][j].copy()
    ctx.new_global_state = out

    variant = "Krum" if m == 1 else f"Multi-Krum (m={m})"
    print(f"  [KRUM] {variant} round={ctx.round} n={n} f={f} "
          f"selected={top_m_idx.tolist()} scores_min={scores.min():.4g} "
          f"scores_max={scores.max():.4g}")
