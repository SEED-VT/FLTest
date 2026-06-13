"""Reference backend: a dependency-light pure-PyTorch FedAvg simulator.

It is the differential-testing *oracle* — a transparent, deterministic implementation
that other frameworks are compared against. It also exposes the richest hook surface
(it can hand a live ``nn.Module`` and gradients to hooks), so attacks like DLG are
easiest to develop here.
"""

from fltest.frameworks.reference.adapter import ReferenceAdapter

__all__ = ["ReferenceAdapter"]
