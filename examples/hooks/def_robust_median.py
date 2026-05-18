"""Coordinate-wise median / trimmed-mean Byzantine-robust aggregation.

Reference:
  Yin, Chen, Ramchandran, Bartlett -- "Byzantine-Robust Distributed
  Learning: Towards Optimal Statistical Rates" (ICML 2018).
  https://arxiv.org/abs/1803.01498

Two variants, selectable via the MEDIAN_VARIANT constant:
  "median"       -- per-coordinate median across all client updates.
                    Robust up to floor((n - 1) / 2) Byzantine clients.
  "trimmed_mean" -- per-coordinate mean after dropping the
                    MEDIAN_TRIM_K * n largest AND smallest values
                    at that coordinate.

These reducers are UNWEIGHTED -- they do not consume Flower's
per-client num_examples. That is standard for coordinate-wise
robust aggregation (the algorithm is defined on equally-weighted
samples) but it means that under unequal partitions the global
model trajectory will differ from weighted FedAvg.

Operates on full client models, not deltas. Equivalent because all
clients in a round start from the same global state.

Load via:
  export FLTEST_HOOKS=examples/hooks/def_robust_median
  poetry run python fltest/main.py
"""

import numpy as np

from fltest.core import hooks


# Configuration
MEDIAN_VARIANT = "median"     # "median" | "trimmed_mean"
# MEDIAN_VARIANT = "trimmed_mean"
MEDIAN_TRIM_K = 0.1           # fraction trimmed from each side for trimmed_mean


def _flatten(update):
    shapes = [a.shape for a in update]
    dtypes = [a.dtype for a in update]
    flat = np.concatenate([a.ravel().astype(np.float64) for a in update])
    return flat, shapes, dtypes


def _unflatten(flat, shapes, dtypes):
    out = []
    offset = 0
    for shape, dtype in zip(shapes, dtypes):
        size = int(np.prod(shape)) if shape else 1
        out.append(flat[offset:offset + size].reshape(shape).astype(dtype))
        offset += size
    return out


def _coordinate_median(stack):
    return np.median(stack, axis=0)


def _coordinate_trimmed_mean(stack, trim_count):
    sorted_stack = np.sort(stack, axis=0)
    trimmed = sorted_stack[trim_count: stack.shape[0] - trim_count]
    return trimmed.mean(axis=0)


@hooks.on_aggregate
def robust_median_aggregate(ctx):
    if not ctx.updates_and_weights:
        return

    updates = [u for u, _ in ctx.updates_and_weights]
    n = len(updates)

    flats_shapes_dtypes = [_flatten(u) for u in updates]
    flats = [fsd[0] for fsd in flats_shapes_dtypes]
    shapes = flats_shapes_dtypes[0][1]
    dtypes = flats_shapes_dtypes[0][2]
    stack = np.stack(flats)

    if MEDIAN_VARIANT == "median":
        agg_flat = _coordinate_median(stack)
        tag = f"median round={ctx.round} n={n}"
    elif MEDIAN_VARIANT == "trimmed_mean":
        trim_count = int(MEDIAN_TRIM_K * n)
        if 2 * trim_count >= n:
            raise ValueError(
                f"[ROBUST] trimmed_mean requires 2*trim_count < n "
                f"(got trim_count={trim_count}, n={n}, k={MEDIAN_TRIM_K}). "
                "Lower MEDIAN_TRIM_K or add more clients."
            )
        agg_flat = _coordinate_trimmed_mean(stack, trim_count)
        tag = f"trimmed_mean(k={MEDIAN_TRIM_K},trim={trim_count}) round={ctx.round} n={n}"
    else:
        raise ValueError(
            f"[ROBUST] unknown MEDIAN_VARIANT={MEDIAN_VARIANT!r}; "
            "must be 'median' or 'trimmed_mean'."
        )

    ctx.new_global_state = _unflatten(agg_flat, shapes, dtypes)
    print(f"  [ROBUST] {tag}")
