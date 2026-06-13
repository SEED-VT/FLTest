"""Built-in metric listeners.

``accuracy`` and ``loss`` are emitted directly by the backends, so they are registered
here as no-op placeholders (declaring them in a config is always valid). Richer metrics
that need extra evaluation are real listeners.
"""

from __future__ import annotations

from statistics import mean
from typing import Optional

import torch

from fltest.core.hook_context import HookContext
from fltest.core.registry import register_metric
from fltest.data.models import get_model
from fltest.data.utils import load_ndarrays_into
from fltest.metrics.base import MetricListenerBaseClass


@register_metric("accuracy")
class AccuracyListener(MetricListenerBaseClass):
    """No-op: backends emit ``accuracy`` each round. Present for config validation."""

    HOOKS = ()


@register_metric("loss")
class LossListener(MetricListenerBaseClass):
    """No-op: backends emit ``loss`` each round. Present for config validation."""

    HOOKS = ()


@register_metric("per_client")
class PerClientAccuracyListener(MetricListenerBaseClass):
    """Personalized evaluation: accuracy of the final global model on each client's data.

    Reports min/mean accuracy across clients (representation disparity), which the
    proposal notes is rarely measured (only ~4% of surveyed works). Requires the backend
    to expose ``dist_dict`` (client loaders) — it caches them at ``on_data_distribute``.
    """

    HOOKS = ("on_data_distribute", "after_simulation")

    def __init__(self, **params):
        super().__init__(**params)
        self._client_loaders = None

    def on_data_distribute(self, ctx: HookContext) -> None:
        # Snapshot client loaders for personalized eval at the end of the run.
        if ctx.dist_dict is not None:
            self._client_loaders = dict(ctx.dist_dict)

    @torch.no_grad()
    def after_simulation(self, ctx: HookContext) -> None:
        loaders = self._client_loaders
        if not loaders or ctx.global_state is None or ctx.cfg is None:
            return
        spec = ctx.cfg
        model = get_model(
            spec.model_name, spec.model_cache_path, channels=spec.channels,
            num_classes=spec.num_classes, deterministic=False,
        ).to(spec.device)
        load_ndarrays_into(model, ctx.global_state)
        model.eval()

        accs = []
        for cid in range(spec.num_clients):
            loader = loaders.get(cid)
            if loader is None:
                continue
            correct, total = 0, 0
            for batch in loader:
                images = batch["img"].to(spec.device)
                labels = batch["label"].to(spec.device)
                preds = torch.argmax(model(images), dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
            if total:
                accs.append(correct / total)
        if accs:
            ctx.record(per_client_acc_mean=mean(accs), per_client_acc_min=min(accs))
