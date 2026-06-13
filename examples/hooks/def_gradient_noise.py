"""Gradient-perturbation defense as a loadable hook file (slide 26).

Run alongside the DLG attack to blunt the reconstruction:

    export FLTEST_HOOKS=examples/hooks/atk_dlg,examples/hooks/def_gradient_noise
    fltest run examples/configs/dlg.yaml

Order matters: the attack hook is listed first so it reconstructs from the *unprotected*
gradient at before_client_train, then this defense clips+noises the update at
after_client_train before it would be shared.
"""

from fltest.core import hooks
from fltest.defenses.gradient_noise import GradientNoiseDefense

_defense = GradientNoiseDefense(clip_norm=1.0, sigma=0.1)


@hooks.after_client_train
def add_gradient_noise(ctx):
    _defense.after_client_train(ctx)
