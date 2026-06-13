"""Metric listeners (the proposal's ``FrameworkMetricListener``), as composable hooks.

Core metrics (``accuracy``, ``loss``) are produced directly by every backend. Additional
listeners registered here add evaluation dimensions — e.g. per-client (personalized)
accuracy, which the proposal flags as commonly missing (Pitfall-3).
"""

from fltest.metrics.base import MetricListenerBaseClass

from fltest.metrics import listeners as _listeners  # noqa: F401

__all__ = ["MetricListenerBaseClass"]
