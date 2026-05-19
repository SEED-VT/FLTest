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


def _validate_layout(updates):
    ref_shapes = [a.shape for a in updates[0]]
    ref_dtypes = [a.dtype for a in updates[0]]
    for k, upd in enumerate(updates[1:], start=1):
        if len(upd) != len(updates[0]):
            raise ValueError(
                f"[ROBUST] client {k} has {len(upd)} tensors; "
                f"expected {len(updates[0])} (matching client 0)."
            )
        for j, (a, ref_s, ref_d) in enumerate(zip(upd, ref_shapes, ref_dtypes)):
            if a.shape != ref_s:
                raise ValueError(
                    f"[ROBUST] client {k} tensor {j} has shape {a.shape}; "
                    f"expected {ref_s}."
                )
            if a.dtype != ref_d:
                raise ValueError(
                    f"[ROBUST] client {k} tensor {j} has dtype {a.dtype}; "
                    f"expected {ref_d}."
                )


def _reduce_coordinate(stack_2d, variant, trim_count):
    """Per-coordinate reduction across a (n, D) stack."""
    if variant == "median":
        return np.median(stack_2d, axis=0)
    sorted_stack = np.sort(stack_2d, axis=0)
    return sorted_stack[trim_count: stack_2d.shape[0] - trim_count].mean(axis=0)


@hooks.on_aggregate
def robust_median_aggregate(ctx):
    if not ctx.updates_and_weights:
        return

    updates = [u for u, _ in ctx.updates_and_weights]
    n = len(updates)
    _validate_layout(updates)

    if MEDIAN_VARIANT == "median":
        trim_count = 0
        tag = f"median round={ctx.round} n={n}"
    elif MEDIAN_VARIANT == "trimmed_mean":
        trim_count = int(MEDIAN_TRIM_K * n)
        if 2 * trim_count >= n:
            raise ValueError(
                f"[ROBUST] trimmed_mean requires 2*trim_count < n "
                f"(got trim_count={trim_count}, n={n}, k={MEDIAN_TRIM_K}). "
                "Lower MEDIAN_TRIM_K or add more clients."
            )
        tag = f"trimmed_mean(k={MEDIAN_TRIM_K},trim={trim_count}) round={ctx.round} n={n}"
    else:
        raise ValueError(
            f"[ROBUST] unknown MEDIAN_VARIANT={MEDIAN_VARIANT!r}; "
            "must be 'median' or 'trimmed_mean'."
        )

    # Aggregate per-tensor. Floating tensors get coordinate-wise reduction;
    # integer tensors (e.g. BN num_batches_tracked) pass through from client 0
    # since coordinate median/mean of small integers would silently truncate.
    out = []
    for j, ref in enumerate(updates[0]):
        if np.issubdtype(ref.dtype, np.floating):
            stack_2d = np.stack(
                [updates[i][j].astype(np.float64).ravel() for i in range(n)]
            )
            reduced = _reduce_coordinate(stack_2d, MEDIAN_VARIANT, trim_count)
            out.append(reduced.reshape(ref.shape).astype(ref.dtype))
        else:
            out.append(updates[0][j].copy())
    ctx.new_global_state = out

    print(f"  [ROBUST] {tag}")
