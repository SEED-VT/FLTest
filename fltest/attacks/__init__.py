"""Attack (threat model) plugins, implemented as composable hooks.

Importing this package registers all built-in attacks into
:data:`fltest.core.registry.ATTACKS` so they are usable by name from a config.
"""

from fltest.attacks.base import ThreatModelBaseClass

# Register built-ins on import.
from fltest.attacks import label_flip as _label_flip  # noqa: F401
from fltest.attacks import sign_flip as _sign_flip  # noqa: F401
from fltest.attacks import gaussian as _gaussian  # noqa: F401
from fltest.attacks import data_poison_backdoor as _backdoor  # noqa: F401
from fltest.attacks import dlg as _dlg  # noqa: F401

__all__ = ["ThreatModelBaseClass"]
