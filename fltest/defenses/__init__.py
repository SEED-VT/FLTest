"""Defense / PPFL-technique plugins, implemented as composable hooks.

Importing this package registers all built-in defenses into
:data:`fltest.core.registry.DEFENSES`.
"""

from fltest.defenses.base import PPFLBaseClass

from fltest.defenses import gradient_noise as _gradient_noise  # noqa: F401
from fltest.defenses import norm_clip as _norm_clip  # noqa: F401
from fltest.defenses import krum as _krum  # noqa: F401
from fltest.defenses import trimmed_mean as _trimmed_mean  # noqa: F401
from fltest.defenses import median as _median  # noqa: F401

__all__ = ["PPFLBaseClass"]
