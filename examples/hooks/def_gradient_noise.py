"""Gradient Perturbation Defense (DP-inspired, NOT formal DP-SGD).

Adds Gaussian noise to the client's model update after training, before it is sent to the server. 
This prevents gradient inversion attacks like DLG from reconstructing private training data.

IMPORTANT NOTE: 

This is NOT true DP-SGD (Abadi et al., 2016). 
Formal DP-SGD requires per-example gradient clipping during training and a privacy accountant to track (epsilon, delta). 
This hook runs after training and can only clip the aggregated update, not individual per-example gradients. 
To implement formal DP-SGD, the training loop itself (fl_testing/frameworks/models.py:train()) would need to be modified. 
We can use the Opacus library to implement formal DP-SGD.

What this does: clip total update norm + add Gaussian noise.
Effect: effective defense against DLG in practice, but without formal privacy guarantees.

Load via:
  export FLTEST_HOOKS=examples/hooks/atk_dlg,examples/hooks/def_gradient_noise
  poetry run python fltest/main.py
"""

import numpy as np

from fltest.core import hooks

# Configuration
DEFENSE_NOISE_SIGMA = 0.001   # Noise std dev (higher = more private, less accurate)
DEFENSE_CLIP_NORM = 1.0       # Max gradient norm (clip before adding noise)

# Defense hook

@hooks.after_client_train
def add_gradient_noise(ctx):
    """Add DP noise to client gradients before sending to server."""
    if ctx.client_update is None:
        return

    # Clip total update norm (NOT per-example; see docstring)
    total_norm = np.sqrt(sum(np.sum(p ** 2) for p in ctx.client_update))
    clip_factor = min(1.0, DEFENSE_CLIP_NORM / (total_norm + 1e-8))

    noisy_update = []
    for param in ctx.client_update:
        clipped = param * clip_factor
        noise = np.random.normal(0, DEFENSE_NOISE_SIGMA, param.shape).astype(param.dtype)
        noisy_update.append(clipped + noise)

    ctx.client_update = noisy_update
    print(f"  [DEF] Gradient noise: sigma={DEFENSE_NOISE_SIGMA}, norm={total_norm:.4f}, clip={clip_factor:.4f}")
