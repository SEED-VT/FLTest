"""A Little Is Enough, backdoor variant: a trigger planted inside the honest spread.

Baruch, Baruch and Goldberg, *A Little Is Enough: Circumventing Defenses For Distributed
Learning* (NeurIPS 2019), Section 3.4 and Algorithm 4. The convergence attack in
``little_is_enough`` moves every parameter a fixed ``z`` deviations; this one spends the
same budget on a goal. Starting from the benign mean ``mu``, the attacker trains on images
stamped with a trigger and relabelled to ``target_label``, then clamps every parameter back
into ``mu +/- z * sigma``:

``p_mal = max(mu - z * sigma, min(v, mu + z * sigma))``

so the crafted update stays inside the range a distance-based defense accepts, while the
parameters inside that range are chosen to carry the backdoor. Every attacker submits it.

The defaults follow the paper's pattern experiment (Section 4.2): each round the attacker
samples ``samples`` of its own images, sets the upper-left ``patch_size`` x ``patch_size``
pixels to maximal intensity, labels them ``target_label`` and trains for ``epochs`` passes.
The FLDetector paper (Zhang et al., KDD 2022) evaluates this attack with the same trigger
and target label 0.

``alpha`` weights the paper's Equation 3, ``alpha * l_backdoor + (1 - alpha) * l_delta``,
where Equation 4's ``l_delta`` sums each parameter's squared distance from ``mu`` in units
of ``max(z * sigma, 1e-5)``. That term is stiff: its curvature is ``1 / (z * sigma)^2``,
and honest clients agree so closely that plain SGD on it diverges at any usable learning
rate, so it is minimised exactly with a proximal step after each gradient step. Summed
over every parameter it also outweighs the batch-mean backdoor loss so heavily that the
optimum barely leaves ``mu``, and the backdoor never forms. The default ``alpha = 1``
therefore leaves "stay close to mu" to the clamp, which is exactly what the clamp is for;
set ``alpha < 1`` to reproduce Equation 3 as written.

``attack_success_rate`` is recorded every round: the share of test images, their true
label not already ``target_label``, that the global model sends to ``target_label`` once
the trigger is stamped on. Threat model and backend support are those of
``little_is_enough``: a full-knowledge attacker acting at ``before_aggregate``, on the
reference and Flower backends.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from fltest.attacks.little_is_enough import LittleIsEnoughAttack
from fltest.core.hook_context import HookContext
from fltest.core.registry import register_attack
from fltest.data.models import LOSS_FUNCTIONS, OPTIMIZERS, get_model
from fltest.data.utils import load_ndarrays_into, state_dict_to_ndarrays


def _stamp(images: torch.Tensor, size: int, value: float) -> torch.Tensor:
    """Set the upper-left ``size`` x ``size`` pixels of each image to ``value``."""
    out = images.clone()
    out[..., :size, :size] = value
    return out


@register_attack("little_is_enough_backdoor")
class LittleIsEnoughBackdoorAttack(LittleIsEnoughAttack):
    """Train a backdoor from the benign mean, then clamp it into ``mu +/- z * sigma``."""

    HOOKS = ("on_data_distribute", "before_aggregate", "after_round")

    def __init__(self, z: Optional[float] = None, alpha: float = 1.0, epochs: int = 5,
                 samples: int = 1000, target_label: int = 0, patch_size: int = 5,
                 patch_value: float = 1.0, **params):
        super().__init__(z=z, **params)
        if not 0.0 < alpha <= 1.0:
            raise ValueError("little_is_enough_backdoor alpha must be in (0, 1]")
        if epochs < 1 or samples < 1 or patch_size < 1:
            raise ValueError("little_is_enough_backdoor epochs, samples and patch_size must be positive")
        self.alpha = float(alpha)
        self.epochs = int(epochs)
        self.samples = int(samples)
        self.target_label = int(target_label)
        self.patch_size = int(patch_size)
        self.patch_value = float(patch_value)
        self._pool = None   # the attackers' own images, pooled
        self._net = None    # scratch model the backdoor is trained in, built once

    def on_data_distribute(self, ctx: HookContext) -> None:
        """Pool the attackers' training images; the backdoor is trained on these."""
        images = []
        for cid, loader in sorted((ctx.dist_dict or {}).items()):
            if not self.targets(cid):
                continue
            dataset = getattr(loader, "dataset", None)
            # A sequential read draws nothing from the global RNG, so honest clients shuffle
            # exactly as they would in a run without this attack.
            batches = DataLoader(dataset, batch_size=512) if dataset is not None else loader
            for batch in batches:
                if "img" not in batch:
                    raise ValueError(
                        "little_is_enough_backdoor stamps a trigger onto pixels, so it needs "
                        "an image dataset. Use little_is_enough on text."
                    )
                images.append(torch.as_tensor(batch["img"]))
        self._pool = torch.cat(images) if images else None

    def _network(self, spec):
        if self._net is None:
            # Building a model draws its initial weights from the global RNG. Fork it so the
            # attacker's scratch model leaves the honest clients' randomness untouched.
            with torch.random.fork_rng(devices=[]):
                self._net = get_model(
                    spec.model_name, spec.model_cache_path, channels=spec.channels,
                    num_classes=spec.num_classes, deterministic=False,
                )
        return self._net.to(spec.device)

    def _craft(self, ctx: HookContext, stats, z: float) -> List[np.ndarray]:
        """Algorithm 4: train from the benign mean on the backdoor, then clamp."""
        if self._pool is None:
            raise ValueError(
                "little_is_enough_backdoor has no attacker images to train on. It reads them at "
                "on_data_distribute, which the reference and Flower backends emit."
            )
        spec, device = ctx.cfg, ctx.cfg.device
        net = self._network(spec)
        start = [reference if mean is None else mean for reference, mean, _ in stats]
        load_ndarrays_into(net, start)

        position = {name: i for i, name in enumerate(net.state_dict())}
        anchors = []
        if self.alpha < 1.0:
            for name, param in net.named_parameters():
                _, mean, deviation = stats[position[name]]
                anchors.append((
                    param,
                    torch.as_tensor(mean, dtype=param.dtype, device=device),
                    torch.as_tensor(np.maximum(z * deviation, 1e-5), dtype=param.dtype, device=device),
                ))

        generator = torch.Generator().manual_seed(int(spec.seed) + int(ctx.round or 0))
        pick = torch.randperm(len(self._pool), generator=generator)[: self.samples]
        images = _stamp(self._pool[pick], self.patch_size, self.patch_value)
        labels = torch.full((len(pick),), self.target_label, dtype=torch.long)

        lr = spec.client_lr
        optimizer = OPTIMIZERS[spec.optimizer](net.parameters(), lr=lr)
        criterion = LOSS_FUNCTIONS[spec.loss_fn]()
        net.train()
        for _ in range(self.epochs):
            order = torch.randperm(len(labels), generator=generator)
            for first in range(0, len(order), spec.client_batch_size):
                batch = order[first:first + spec.client_batch_size]
                optimizer.zero_grad()
                loss = self.alpha * criterion(net(images[batch].to(device)), labels[batch].to(device))
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    # Proximal step for (1 - alpha) * l_delta: its exact minimiser from here,
                    # which stays stable however small z * sigma is.
                    for param, anchor, scale in anchors:
                        c = 2.0 * lr * (1.0 - self.alpha) / scale ** 2
                        param.copy_((param + c * anchor) / (1.0 + c))

        return LittleIsEnoughCroppedBackdoorAttack._crop(
            state_dict_to_ndarrays(net.state_dict()), stats, z)

    @torch.no_grad()
    def after_round(self, ctx: HookContext) -> None:
        self._score_backdoor(ctx)

    def _score_backdoor(self, ctx: HookContext) -> None:
        if ctx.global_state is None or ctx.test_data is None or ctx.cfg is None:
            return
        spec = ctx.cfg
        model = ctx.model
        if model is None:
            model = self._network(spec)
            load_ndarrays_into(model, ctx.global_state)
        model.eval()
        hit, total = 0, 0
        for batch in ctx.test_data:
            if "img" not in batch:
                return
            labels = batch["label"].to(spec.device)
            keep = labels != self.target_label
            if not keep.any():
                continue
            images = _stamp(batch["img"].to(spec.device)[keep], self.patch_size, self.patch_value)
            hit += int((model(images).argmax(dim=1) == self.target_label).sum().item())
            total += int(keep.sum().item())
        if total:
            ctx.record(attack_success_rate=hit / total)


@register_attack("little_is_enough_cropped_backdoor")
class LittleIsEnoughCroppedBackdoorAttack(LittleIsEnoughBackdoorAttack):
    """Each attacker trains its own Scaling-attack backdoor, then crops it into range.

    The FLDetector paper (Zhang et al., KDD 2022) evaluates "A Little Is Enough" this way:
    the malicious updates "are first computed following the Scaling Attack", and then "the
    attacker crops the model updates to be in certain ranges". Here that is, per attacker
    and per round: start from the global model, train on the attacker's own images plus a
    trigger-stamped duplicate of each, labelled ``target_label`` (the Scaling attack's
    augmentation, with the paper's scaling factor of 1), then clamp every parameter into
    the honest ``mu +/- z * sigma``.

    ``little_is_enough_backdoor`` follows Algorithm 4 of the ALIE paper instead: one update,
    trained from the honest mean ``mu`` and shared by every attacker. The two differ in
    what the malicious update is anchored to, which is what a consistency detector such as
    FLDetector reads. Local training uses the run's own ``client_epochs``, ``client_lr``,
    ``client_batch_size`` and ``optimizer``, as an honest client's does.
    """

    def __init__(self, z: Optional[float] = None, target_label: int = 0, patch_size: int = 5,
                 patch_value: float = 1.0, **params):
        super().__init__(z=z, target_label=target_label, patch_size=patch_size,
                         patch_value=patch_value, **params)
        self._shards = {}  # attacker client id -> (images, labels)

    def on_data_distribute(self, ctx: HookContext) -> None:
        """Keep each attacker's own images and labels; every attacker trains on its own."""
        self._shards = {}
        for cid, loader in sorted((ctx.dist_dict or {}).items()):
            if not self.targets(cid):
                continue
            dataset = getattr(loader, "dataset", None)
            batches = DataLoader(dataset, batch_size=512) if dataset is not None else loader
            images, labels = [], []
            for batch in batches:
                if "img" not in batch:
                    raise ValueError(
                        "little_is_enough_cropped_backdoor stamps a trigger onto pixels, so it "
                        "needs an image dataset. Use little_is_enough on text."
                    )
                images.append(torch.as_tensor(batch["img"]))
                labels.append(torch.as_tensor(batch["label"]))
            if images:
                self._shards[cid] = (torch.cat(images), torch.cat(labels))

    def _craft_each(self, ctx: HookContext, stats, z: float, attackers) -> dict:
        if ctx.global_state is None:
            raise ValueError("little_is_enough_cropped_backdoor needs the round's global model")
        records = ctx.client_submissions
        aligned = records is not None and len(records) == len(ctx.updates_and_weights)
        crafted = {}
        for pos in attackers:
            cid = records[pos].client_id if aligned and records[pos].client_id is not None else pos
            if cid not in self._shards:
                raise ValueError(
                    f"little_is_enough_cropped_backdoor has no training data for attacker {cid}. "
                    "It reads it at on_data_distribute, which the reference and Flower backends emit."
                )
            crafted[pos] = self._crop(self._poisoned_training(ctx, cid), stats, z)
        return crafted

    def _poisoned_training(self, ctx: HookContext, cid: int) -> List[np.ndarray]:
        """The Scaling attack's local update: own data plus stamped duplicates, from the global model."""
        spec, device = ctx.cfg, ctx.cfg.device
        net = self._network(spec)
        load_ndarrays_into(net, ctx.global_state)
        images, labels = self._shards[cid]
        images = torch.cat([images, _stamp(images, self.patch_size, self.patch_value)])
        labels = torch.cat([labels, torch.full_like(labels, self.target_label)])

        generator = torch.Generator().manual_seed(
            int(spec.seed) * 100_003 + int(ctx.round or 0) * 1_009 + int(cid))
        optimizer = OPTIMIZERS[spec.optimizer](net.parameters(), lr=spec.client_lr)
        criterion = LOSS_FUNCTIONS[spec.loss_fn]()
        net.train()
        for _ in range(spec.client_epochs):
            order = torch.randperm(len(labels), generator=generator)
            for first in range(0, len(order), spec.client_batch_size):
                batch = order[first:first + spec.client_batch_size]
                optimizer.zero_grad()
                criterion(net(images[batch].to(device)), labels[batch].to(device)).backward()
                optimizer.step()
        return state_dict_to_ndarrays(net.state_dict())

    @staticmethod
    def _crop(trained: List[np.ndarray], stats, z: float) -> List[np.ndarray]:
        """Clamp each floating-point layer into the honest ``mu +/- z * sigma``."""
        cropped = []
        for (reference, mean, deviation), value in zip(stats, trained):
            if mean is None:
                cropped.append(reference.copy())
                continue
            bound = z * deviation
            cropped.append(
                np.clip(value.astype(np.float64), mean - bound, mean + bound)
                .astype(reference.dtype, copy=False)
            )
        return cropped
