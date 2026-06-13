"""Deep Leakage from Gradients (DLG) — gradient-inversion privacy attack.

Reconstructs a victim client's private training batch (images + labels) by optimizing a
dummy batch so its gradient matches the victim's, demonstrating that shared gradients are
invertible — the canonical privacy threat the testbed reproduces (slides 23–25).

Two threat sources:

* ``source="gradient"`` (default): an honest-but-curious party who can read the victim's
  *raw* per-step gradient. Reconstructs at ``before_client_train``. Shows pure invertibility,
  independent of any update-level defense.
* ``source="shared_update"``: a server that only sees the *uploaded* update. Reconstructs at
  ``before_aggregate`` from ``g ≈ (global − update) / lr`` using the **post-defense** update,
  so a gradient-perturbation defense (clip+noise) measurably raises reconstruction error
  (slides 26–27). Most faithful with a single local step (``client_epochs: 1``).

Records reconstruction MSE/PSNR + label-recovery; optionally saves a GT-vs-reconstruction grid.
Best with a smooth-activation model (``model_name: ConvNet``) on ``device: cpu``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from fltest.attacks.base import ThreatModelBaseClass
from fltest.core.hook_context import HookContext
from fltest.core.registry import register_attack
from fltest.data.models import get_model
from fltest.data.utils import load_ndarrays_into, seed_everything


@register_attack("dlg")
class DLGAttack(ThreatModelBaseClass):
    def __init__(
        self,
        target_client: int = 0,
        target_round: int = 1,
        num_images: int = 1,
        iters: int = 300,
        lr: float = 1.0,
        source: str = "gradient",
        save_dir: Optional[str] = None,
        **params,
    ):
        super().__init__(**params)
        self.target_client = target_client
        self.target_round = target_round
        self.num_images = num_images
        self.iters = iters
        self.lr = lr
        self.source = source
        self.save_dir = save_dir
        # capture ground-truth at before_client_train; reconstruct where the threat sees data.
        self.HOOKS = ("before_client_train", "before_aggregate") if source == "shared_update" \
            else ("before_client_train",)
        self._gt = None  # (images, labels) of the victim batch, for MSE evaluation

    # ---- victim-batch capture (always) + raw-gradient reconstruction (gradient mode) ----
    def before_client_train(self, ctx: HookContext) -> None:
        if ctx.client_id != self.target_client or ctx.round != self.target_round:
            return
        if ctx.global_state is None or ctx.client_data is None or ctx.cfg is None:
            return
        model, images, labels = self._setup(ctx)
        self._gt = (images.detach().clone(), labels.detach().clone())
        if self.source == "gradient":
            params = [p for p in model.parameters() if p.requires_grad]
            true_loss = F.cross_entropy(model(images), labels)
            true_grads = [g.detach() for g in torch.autograd.grad(true_loss, params)]
            self._reconstruct(ctx, model, params, true_grads, images, labels)

    # ---- server-side reconstruction from the (post-defense) shared update ----
    def before_aggregate(self, ctx: HookContext) -> None:
        if ctx.round != self.target_round or self._gt is None or ctx.updates_and_weights is None:
            return
        if self.target_client >= len(ctx.updates_and_weights) or ctx.global_state is None:
            return
        spec = ctx.cfg
        images, labels = self._gt
        update = ctx.updates_and_weights[self.target_client][0]
        g = ctx.global_state
        # gradient implied by the uploaded update: update ≈ global - lr * grad
        observed = [torch.as_tensor((gi - ui) / spec.client_lr) for ui, gi in zip(update, g)]
        model = self._build_model(spec)
        load_ndarrays_into(model, g)
        params = [p for p in model.parameters() if p.requires_grad]
        observed = [o.to(images.device) for o in observed]
        self._reconstruct(ctx, model, params, observed, images, labels)

    # ---- shared reconstruction core ----
    def _build_model(self, spec):
        return get_model(
            spec.model_name, spec.model_cache_path, channels=spec.channels,
            num_classes=spec.num_classes, deterministic=False,
        ).to(spec.device)

    def _setup(self, ctx):
        spec = ctx.cfg
        seed_everything(spec.seed)
        model = self._build_model(spec)
        load_ndarrays_into(model, ctx.global_state)
        model.eval()
        batch = next(iter(ctx.client_data))
        images = batch["img"][: self.num_images].to(spec.device).float()
        labels = batch["label"][: self.num_images].to(spec.device).long()
        return model, images, labels

    def _reconstruct(self, ctx, model, params, true_grads, images, labels) -> None:
        spec = ctx.cfg
        device = spec.device
        dummy_data = torch.randn_like(images, requires_grad=True)
        dummy_label = torch.randn(images.size(0), spec.num_classes, device=device, requires_grad=True)
        optimizer = torch.optim.LBFGS([dummy_data, dummy_label], lr=self.lr)

        def closure():
            optimizer.zero_grad()
            pred = model(dummy_data)
            loss = torch.mean(torch.sum(-F.log_softmax(pred, -1) * F.softmax(dummy_label, -1), -1))
            grads = torch.autograd.grad(loss, params, create_graph=True)
            diff = sum(((gr - t) ** 2).sum() for gr, t in zip(grads, true_grads))
            diff.backward()
            return diff

        for _ in range(self.iters):
            optimizer.step(closure)

        with torch.no_grad():
            mse = F.mse_loss(dummy_data, images).item()
            rng = (images.max() - images.min()).item() ** 2
            psnr = float("inf") if mse == 0 else 10.0 * math.log10(rng / mse)
            label_recovery = (torch.argmax(dummy_label, -1) == labels).float().mean().item()

        ctx.record(reconstruction_mse=mse, reconstruction_psnr=psnr, label_recovery=label_recovery)
        ctx.extras["dlg"] = {"mse": mse, "psnr": psnr, "label_recovery": label_recovery,
                             "source": self.source}
        if self.save_dir or getattr(spec, "fw_cache_path", None):
            self._save(images, dummy_data.detach(), spec)

    @staticmethod
    def _save(true_images, recon_images, spec) -> None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception:
            return
        out_dir = Path(spec.fw_cache_path) / "dlg"
        out_dir.mkdir(parents=True, exist_ok=True)

        def to_img(t):
            arr = (t * 0.5 + 0.5).clamp(0, 1).cpu().numpy()
            return arr[0] if arr.shape[0] == 1 else arr.transpose(1, 2, 0)

        n = true_images.size(0)
        fig, axes = plt.subplots(2, n, figsize=(2 * n, 4), squeeze=False)
        for i in range(n):
            axes[0, i].imshow(to_img(true_images[i]), cmap="gray"); axes[0, i].axis("off")
            axes[0, i].set_title("GT")
            axes[1, i].imshow(to_img(recon_images[i]), cmap="gray"); axes[1, i].axis("off")
            axes[1, i].set_title("Recon")
        fig.tight_layout()
        fig.savefig(out_dir / "dlg_reconstruction.png", dpi=120)
        plt.close(fig)
