"""Attack (threat model) plugins, implemented as composable hooks.

Importing this package declares all built-in attacks in
:data:`fltest.core.registry.ATTACKS` so they are usable by name from a config. The
declarations are lazy: each implementing module is imported the first time its attack is
requested, so importing this package does not pull in torch.
"""

from fltest.attacks.base import ThreatModelBaseClass
from fltest.core.registry import ATTACKS

#: registry name -> module that defines and registers it
BUILTIN_ATTACKS = {
    "label_flip": "fltest.attacks.label_flip",
    "sign_flip": "fltest.attacks.sign_flip",
    "gaussian": "fltest.attacks.gaussian",
    "backdoor": "fltest.attacks.data_poison_backdoor",
    "dlg": "fltest.attacks.dlg",
    "membership_inference": "fltest.attacks.membership_inference",
    "model_replacement": "fltest.attacks.model_replacement",
}

for _name, _module in BUILTIN_ATTACKS.items():
    ATTACKS.register_lazy(_name, _module)

__all__ = ["ThreatModelBaseClass", "BUILTIN_ATTACKS"]
