"""DLG gradient-inversion attack as a loadable hook file (slide 24).

Demonstrates the FLTEST_HOOKS mechanism: define attacks logically with the hook decorators,
then load them at runtime without touching any config:

    export FLTEST_HOOKS=examples/hooks/atk_dlg
    fltest run examples/configs/dlg.yaml

This thin wrapper delegates to the built-in, well-tested DLGAttack implementation.
"""

from fltest.attacks.dlg import DLGAttack
from fltest.core import hooks

_attack = DLGAttack(target_client=0, target_round=1, num_images=1, iters=100)


@hooks.before_client_train
def run_dlg_attack(ctx):
    _attack.before_client_train(ctx)
