"""Base class for attacks (threat models).

``ThreatModelBaseClass`` is the overridable interface from the proposal: subclass it,
declare the lifecycle hooks you use in ``HOOKS``, and implement the matching methods.
Each method receives the shared HookContext, so an attack runs unchanged across backends.
"""

from __future__ import annotations

from fltest.core.plugin import HookPlugin


class ThreatModelBaseClass(HookPlugin):
    """Base class for all attacks. See :class:`fltest.core.plugin.HookPlugin`."""

    #: Human-readable category, used in reports.
    category: str = "attack"
