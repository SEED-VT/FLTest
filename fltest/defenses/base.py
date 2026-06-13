"""Base class for defenses / PPFL techniques.

``PPFLBaseClass`` is the proposal's overridable PPFL interface. Two flavors of defense
compose through the same hooks:

* **Client-side perturbation** (DP-style: clip + noise) acts at ``after_client_train`` on
  one client's update.
* **Robust aggregation** (Krum / trimmed-mean / median) acts at ``before_aggregate`` by
  replacing the ``updates_and_weights`` list the backend will average.
"""

from __future__ import annotations

from fltest.core.plugin import HookPlugin


class PPFLBaseClass(HookPlugin):
    """Base class for all defenses. See :class:`fltest.core.plugin.HookPlugin`."""

    category: str = "defense"
