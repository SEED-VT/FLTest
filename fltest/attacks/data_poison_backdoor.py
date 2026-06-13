"""Backdoor data-poisoning attack with a pixel-patch trigger.

Attacker clients stamp a small bright patch onto a fraction of their images and relabel
them to ``target_label``. The global model thus learns: *trigger present => target_label*.
At each round end the attack measures **attack success rate (ASR)** — the fraction of a
triggered test set the global model classifies as the target — and records it as a metric,
giving the testbed the robustness signal the proposal calls for.
"""

from __future__ import annotations

import torch

from fltest.attacks.base import ThreatModelBaseClass
from fltest.core.hook_context import HookContext
from fltest.core.registry import register_attack
from fltest.data.models import get_model
from fltest.data.utils import load_ndarrays_into


def _stamp(images: torch.Tensor, size: int, value: float) -> torch.Tensor:
    """Stamp a ``size``x``size`` bright square into the bottom-right corner of each image."""
    out = images.clone()
    out[..., -size:, -size:] = value
    return out


class _BackdoorLoader:
    def __init__(self, base, frac: float, size: int, value: float, target: int):
        self._base = base
        self.dataset = getattr(base, "dataset", None)
        self._frac, self._size, self._value, self._target = frac, size, value, target

    def __len__(self):
        return len(self._base)

    def __iter__(self):
        for batch in self._base:
            images = batch["img"].clone()
            labels = torch.as_tensor(batch["label"]).clone()
            n = labels.size(0)
            k = int(self._frac * n)
            if k > 0:
                images[:k] = _stamp(images[:k], self._size, self._value)
                labels[:k] = self._target
            new = dict(batch)
            new["img"], new["label"] = images, labels
            yield new


@register_attack("backdoor")
class BackdoorAttack(ThreatModelBaseClass):
    HOOKS = ("before_client_train", "after_round")

    def __init__(self, target_label: int = 0, infection_rate: float = 0.3,
                 patch_size: int = 4, patch_value: float = 1.0, **params):
        super().__init__(**params)
        self.target_label = target_label
        self.infection_rate = infection_rate
        self.patch_size = patch_size
        self.patch_value = patch_value

    def before_client_train(self, ctx: HookContext) -> None:
        if not self.targets(ctx.client_id) or ctx.client_data is None:
            return
        ctx.client_data = _BackdoorLoader(
            ctx.client_data, self.infection_rate, self.patch_size, self.patch_value, self.target_label
        )

    @torch.no_grad()
    def after_round(self, ctx: HookContext) -> None:
        if ctx.global_state is None or ctx.test_data is None or ctx.cfg is None:
            return
        spec = ctx.cfg
        model = ctx.model
        if model is None:
            model = get_model(
                spec.model_name, spec.model_cache_path, channels=spec.channels,
                num_classes=spec.num_classes, deterministic=False,
            ).to(spec.device)
            load_ndarrays_into(model, ctx.global_state)
        model.eval()

        hit, total = 0, 0
        for batch in ctx.test_data:
            images = _stamp(batch["img"].to(spec.device), self.patch_size, self.patch_value)
            labels = batch["label"].to(spec.device)
            # Exclude samples whose true label is already the target.
            mask = labels != self.target_label
            if mask.sum() == 0:
                continue
            preds = torch.argmax(model(images[mask]), dim=1)
            hit += (preds == self.target_label).sum().item()
            total += int(mask.sum().item())
        if total:
            ctx.record(attack_success_rate=hit / total)
