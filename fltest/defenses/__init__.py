"""Defense / PPFL-technique plugins, implemented as composable hooks.

Importing this package declares all built-in defenses in
:data:`fltest.core.registry.DEFENSES`. The declarations are lazy: each implementing
module is imported the first time its defense is requested.
"""

from fltest.core.registry import DEFENSES
from fltest.defenses.base import PPFLBaseClass

#: registry name -> module that defines and registers it
BUILTIN_DEFENSES = {
    "gradient_noise": "fltest.defenses.gradient_noise",
    "norm_clip": "fltest.defenses.norm_clip",
    "krum": "fltest.defenses.krum",
    "trimmed_mean": "fltest.defenses.trimmed_mean",
    "median": "fltest.defenses.median",
    "fldetector": "fltest.defenses.fldetector",
}

for _name, _module in BUILTIN_DEFENSES.items():
    DEFENSES.register_lazy(_name, _module)

__all__ = ["PPFLBaseClass", "BUILTIN_DEFENSES"]
