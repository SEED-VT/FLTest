"""Label-flipping data-poisoning attack.

Attacker clients train on relabeled data. By default labels are rotated
``y -> (y + shift) % num_classes``; a fixed ``{src: dst}`` mapping can be given instead.
Acts at ``before_client_train`` by wrapping the client's DataLoader, so it is identical
across backends (they all pass a ``{"img","label"}`` loader through ``ctx.client_data``).
"""

from __future__ import annotations

import torch

from fltest.attacks.base import ThreatModelBaseClass
from fltest.core.hook_context import HookContext
from fltest.core.registry import register_attack


class _RelabelLoader:
    """Wraps a DataLoader and relabels each batch on the fly."""

    def __init__(self, base, num_classes: int, shift: int, mapping: dict | None):
        self._base = base
        self.dataset = getattr(base, "dataset", None)
        self._num_classes = num_classes
        self._shift = shift
        self._mapping = mapping

    def __len__(self):
        return len(self._base)

    def __iter__(self):
        for batch in self._base:
            labels = torch.as_tensor(batch["label"])
            if self._mapping:
                flipped = labels.clone()
                for src, dst in self._mapping.items():
                    flipped[labels == int(src)] = int(dst)
            else:
                flipped = (labels + self._shift) % self._num_classes
            batch = dict(batch)
            batch["label"] = flipped.to(labels.device).type_as(labels)
            yield batch


@register_attack("label_flip")
class LabelFlipAttack(ThreatModelBaseClass):
    """Relabel an attacker client's training data before local training."""

    HOOKS = ("before_client_train",)

    def __init__(self, shift: int = 1, mapping: dict | None = None, **params):
        super().__init__(**params)
        self.shift = shift
        self.mapping = mapping

    def before_client_train(self, ctx: HookContext) -> None:
        if not self.targets(ctx.client_id) or ctx.client_data is None:
            return
        num_classes = getattr(ctx.cfg, "num_classes", 10)
        ctx.client_data = _RelabelLoader(ctx.client_data, num_classes, self.shift, self.mapping)
