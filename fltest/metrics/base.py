"""Base class for metric listeners."""

from __future__ import annotations

from fltest.core.plugin import HookPlugin


class MetricListenerBaseClass(HookPlugin):
    """Base for metric listeners. Implement lifecycle methods and ``ctx.record(...)``."""

    category: str = "metric"
