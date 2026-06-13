"""FL framework adapters.

Each backend implements :class:`fltest.frameworks.base.FrameworkAdapter` and registers
itself by name. Importing this package triggers registration of the always-available
backends (reference, flower). Heavy/optional backends (nvflare, pfl) register lazily and
only if their dependencies are installed.
"""

from fltest.frameworks.base import FrameworkAdapter, RunResult, get_adapter

# Always-available, dependency-light backends.
from fltest.frameworks import reference as _reference  # noqa: F401
from fltest.frameworks import flower as _flower  # noqa: F401


def _try_register_optional() -> None:
    """Register heavy optional backends if their deps import cleanly."""
    for module in ("fltest.frameworks.nvflare", "fltest.frameworks.pfl"):
        try:
            __import__(module)
        except Exception:
            # Optional extra not installed; skip silently. `fltest run` reports a clear
            # error only if a config actually requests the missing backend.
            pass


_try_register_optional()

__all__ = ["FrameworkAdapter", "RunResult", "get_adapter"]
