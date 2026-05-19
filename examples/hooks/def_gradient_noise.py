"""Gradient Perturbation Defense (DP-inspired, NOT formal DP-SGD).

Adds Gaussian noise to the client's UPDATE DELTA (parameters_out -
global_state_before) and optionally clips the delta's L2 norm. This is a
heuristic defense against gradient-inversion attacks (DLG) and a weak
deterrent against the Bagdasaryan model-replacement attack: the clip caps
how much the malicious update can dominate FedAvg, and the noise breaks the
attacker's ability to encode a precise backdoor signal.

Operates on the DELTA, not the full returned model. v1 of this hook clipped
the full model parameter vector, which under ConvNet (norm ~40) shrank the
model by ~40x per round and collapsed training to NaN within 2 rounds.

IMPORTANT NOTE:

This is NOT true DP-SGD (Abadi et al., 2016).
Formal DP-SGD requires per-example gradient clipping during training and a
privacy accountant to track (epsilon, delta). This hook runs after training
and can only clip the aggregated update, not individual per-example gradients.
To implement formal DP-SGD, the training loop itself
(fl_testing/frameworks/models.py:train()) would need to be modified, e.g.
via the Opacus library.

Load via:
  export FLTEST_HOOKS=examples/hooks/atk_backdoor,examples/hooks/def_gradient_noise
"""

import numpy as np

from fltest.core import hooks


# Configuration
DEFENSE_NOISE_SIGMA = 1e-4    # Noise std dev applied to each delta coordinate
DEFENSE_CLIP_NORM = 5.0       # L2 cap on the per-client delta (NOT the full model)


# Module-level cache: (round, client_id) -> list of pre-train ndarrays.
# Workers and driver run in separate processes under Ray simulation, but this
# hook lives entirely on the worker side so a module global is fine.
_global_state_cache = {}


@hooks.before_client_train
def cache_pretrain_state(ctx):
    if ctx.global_state is None or ctx.client_id is None or ctx.round is None:
        return
    _global_state_cache[(ctx.round, ctx.client_id)] = [
        np.array(a, copy=True) for a in ctx.global_state
    ]


@hooks.after_client_train
def clip_and_noise_delta(ctx):
    if ctx.client_update is None:
        return
    pre = _global_state_cache.pop((ctx.round, ctx.client_id), None)
    if pre is None:
        # No cached state -- can't compute the delta. Skip rather than corrupt.
        print(f"  [DEF] WARNING round={ctx.round} client={ctx.client_id}: no cached pre-train state, defense skipped")
        return

    deltas = []
    for u, g in zip(ctx.client_update, pre):
        if np.issubdtype(g.dtype, np.floating):
            deltas.append(u - g)
        else:
            deltas.append(None)  # integer state (BN num_batches_tracked) -- skip

    # Compute L2 norm over the float-delta portion only.
    total_norm = float(np.sqrt(sum(
        np.sum(d ** 2) for d in deltas if d is not None
    )))
    clip_factor = min(1.0, DEFENSE_CLIP_NORM / (total_norm + 1e-12))

    new_update = []
    for u, g, d in zip(ctx.client_update, pre, deltas):
        if d is None:
            # Integer buffer: pass through the attacker's value unchanged.
            new_update.append(u.copy())
            continue
        clipped = d * clip_factor
        noise = np.random.normal(0, DEFENSE_NOISE_SIGMA, clipped.shape).astype(clipped.dtype)
        new_update.append(g + clipped + noise)
    ctx.client_update = new_update

    print(
        f"  [DEF] round={ctx.round} client={ctx.client_id} "
        f"delta_norm={total_norm:.4f} clip_factor={clip_factor:.4f} "
        f"sigma={DEFENSE_NOISE_SIGMA:g}"
    )
