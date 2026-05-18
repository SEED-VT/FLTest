"""Backdoor / model-replacement attack (Bagdasaryan AISTATS 2020).

Reference:
  Bagdasaryan, Veitch, Hua, Estrin, Shmatikov -- "How To Backdoor
  Federated Learning" (AISTATS 2020).
  https://arxiv.org/abs/1807.00459 ;
  https://github.com/ebagdasa/backdoor_federated_learning

Attack stages:
  1. @before_client_train: at round BACKDOOR_ATTACK_ROUND for client
     BACKDOOR_TARGET_CLIENT, wrap the client's DataLoader with
     _PoisonedLoader. The wrapper applies a fixed pixel-patch trigger
     to BACKDOOR_POISON_RATIO of samples per batch and relabels them
     to BACKDOOR_TARGET_CLASS. Cache the pre-train global state
     (deep-copied numpy) keyed by (round, client_id) for the matching
     after-train delta.
  2. @after_client_train: scale the attacker's update by gamma so it
     dominates FedAvg. With Flower averaging client parameters by
     num_examples and no server learning rate, gamma is
     sum(selected_weights) / attacker_weight. The hook only sees its
     own num_samples and cfg, so "auto" resolves to
     cfg.num_clients * cfg.fraction_fit  (assuming roughly equal
     weights, which the default Dirichlet/IID partition gives).
     Override with a fixed integer if you know the true sum.
  3. @after_round: score the global model on a clean test set AND on
     a triggered test set (every image patched, every label replaced
     with BACKDOOR_TARGET_CLASS). backdoor_acc is the fraction of
     triggered samples the model predicts as TARGET_CLASS -- i.e.
     the attack success rate, not accuracy against original labels.
     Append a row to tmp/backdoor_results/metrics.csv.

Trigger type:
  Pixel pattern only. A real Bagdasaryan-style semantic backdoor
  (e.g. "green cars are relabeled to bird") requires hand-curating
  rare-feature samples and is left to a future iteration.

Gamma caveat:
  The paper's gamma = n / eta formula assumes a server learning rate
  eta. This codebase has no server LR (Flower's FedAvg with
  fraction_fit=1.0 just averages full client models). The
  substitution above is the practical analogue under this setup.

Not implemented:
  - Attacker-specific LR / epochs cannot be reached from hooks
    (would require changing the client/train API).
  - Continuous-poison + constrain-and-scale variants are future
    work (Stage 8).

Load via:
  export FLTEST_HOOKS=examples/hooks/atk_backdoor
  poetry run python fltest/main.py dataset=cifar10 model_name=ConvNet num_rounds=5

Combine with a defense:
  export FLTEST_HOOKS=examples/hooks/atk_backdoor,examples/hooks/def_krum
"""

import csv
import os
from pathlib import Path

import numpy as np
import torch

from fltest.core import hooks
from fl_testing.frameworks.models import get_pytorch_model
from fltest.adapters.flower.utils import set_parameters


# Configuration
BACKDOOR_TARGET_CLIENT = 0    # which client is malicious
BACKDOOR_ATTACK_ROUND = 3     # which round to inject (1-indexed; first FL round is 1)
BACKDOOR_TARGET_CLASS = 2     # adversary's chosen target label
BACKDOOR_PATCH_SIZE = 4       # trigger patch side length, pixels
BACKDOOR_PATCH_VALUE = 1.0    # patch fill value (1.0 = white after normalization)
BACKDOOR_POISON_RATIO = 0.5   # fraction of attacker's batch to poison
BACKDOOR_SCALE_GAMMA = "auto" # "auto" -> num_clients * fraction_fit, or a fixed int

_RESULTS_DIR = Path("tmp/backdoor_results")
_CSV_PATH = _RESULTS_DIR / "metrics.csv"
_CSV_HEADER = [
    "round", "variant", "defense",
    "main_acc", "backdoor_acc",
    "attacker_update_l2", "scaled_update_l2",
]


# Module-level state
_global_state_cache = {}        # (round, client_id) -> list of numpy arrays
_attacker_l2 = {"unscaled": None, "scaled": None}  # latest attacker norms for after_round


def _apply_pixel_trigger(img_tensor):
    """Paint a fixed white square in the top-left corner. Operates on a clone."""
    img = img_tensor.clone()
    img[..., :BACKDOOR_PATCH_SIZE, :BACKDOOR_PATCH_SIZE] = BACKDOOR_PATCH_VALUE
    return img


class _PoisonedLoader:
    """DataLoader wrapper. Round-aware: only poisons on BACKDOOR_ATTACK_ROUND.

    Anti-restack: if `base` is already a _PoisonedLoader, unwrap to its `.base`
    before storing -- otherwise repeated wrapping across rounds would nest.
    """

    def __init__(self, base, current_round, target_class, ratio):
        while isinstance(base, _PoisonedLoader):
            base = base.base
        self.base = base
        self.current_round = current_round
        self.target_class = target_class
        self.ratio = ratio
        self.active = (current_round == BACKDOOR_ATTACK_ROUND)
        self.dataset = getattr(base, "dataset", None)

    def __len__(self):
        return len(self.base)

    def __iter__(self):
        for batch in self.base:
            if not self.active:
                yield batch
                continue
            # Clone before mutating -- DataLoader buffers may be reused.
            img = batch["img"].clone()
            label = batch["label"].clone()
            n = img.shape[0]
            n_poison = max(1, int(round(self.ratio * n)))
            idx = torch.randperm(n)[:n_poison]
            for i in idx.tolist():
                img[i] = _apply_pixel_trigger(img[i])
                label[i] = self.target_class
            new_batch = dict(batch)
            new_batch["img"] = img
            new_batch["label"] = label
            yield new_batch


def _resolve_gamma(ctx):
    if isinstance(BACKDOOR_SCALE_GAMMA, (int, float)):
        return float(BACKDOOR_SCALE_GAMMA)
    if BACKDOOR_SCALE_GAMMA == "auto":
        fraction_fit = getattr(ctx.cfg, "fraction_fit", 1.0)
        return float(ctx.cfg.num_clients) * float(fraction_fit)
    raise ValueError(
        f"BACKDOOR_SCALE_GAMMA must be 'auto' or a number; got {BACKDOOR_SCALE_GAMMA!r}"
    )


def _ensure_csv_header():
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if not _CSV_PATH.exists() or _CSV_PATH.stat().st_size == 0:
        with open(_CSV_PATH, "w", newline="") as f:
            csv.writer(f).writerow(_CSV_HEADER)


def _l2(ndarray_list):
    return float(np.sqrt(sum(np.sum(a ** 2) for a in ndarray_list)))


def _l2_delta(post, pre):
    return float(np.sqrt(sum(np.sum((p - q) ** 2) for p, q in zip(post, pre))))


@hooks.before_client_train
def poison_and_cache(ctx):
    """Poison the attacker's loader on attack round; cache pre-train global state."""
    if ctx.client_id != BACKDOOR_TARGET_CLIENT:
        return
    if ctx.round != BACKDOOR_ATTACK_ROUND:
        return
    if ctx.global_state is not None:
        _global_state_cache[(ctx.round, ctx.client_id)] = [
            np.array(a, copy=True) for a in ctx.global_state
        ]
    if ctx.client_data is not None:
        ctx.client_data = _PoisonedLoader(
            ctx.client_data,
            current_round=ctx.round,
            target_class=BACKDOOR_TARGET_CLASS,
            ratio=BACKDOOR_POISON_RATIO,
        )
        print(
            f"  [BACKDOOR] poisoning client {ctx.client_id} round {ctx.round} "
            f"target_class={BACKDOOR_TARGET_CLASS} ratio={BACKDOOR_POISON_RATIO} "
            f"patch={BACKDOOR_PATCH_SIZE}x{BACKDOOR_PATCH_SIZE}"
        )


@hooks.after_client_train
def scale_update(ctx):
    """Scale the attacker's delta by gamma (model-replacement attack)."""
    if ctx.client_id != BACKDOOR_TARGET_CLIENT:
        return
    if ctx.round != BACKDOOR_ATTACK_ROUND:
        return
    pre = _global_state_cache.pop((ctx.round, ctx.client_id), None)
    if pre is None or ctx.client_update is None:
        return
    gamma = _resolve_gamma(ctx)
    delta = [u - g for u, g in zip(ctx.client_update, pre)]
    unscaled_l2 = _l2(delta)
    scaled_update = [g + gamma * d for g, d in zip(pre, delta)]
    scaled_l2 = _l2_delta(scaled_update, pre)
    ctx.client_update = scaled_update
    _attacker_l2["unscaled"] = unscaled_l2
    _attacker_l2["scaled"] = scaled_l2
    print(
        f"  [BACKDOOR] scaled client {ctx.client_id} round {ctx.round} "
        f"gamma={gamma:g} unscaled_l2={unscaled_l2:.4f} scaled_l2={scaled_l2:.4f}"
    )


@hooks.after_round
def score_round_and_log(ctx):
    """Score main and backdoor accuracy on the global model; append CSV row."""
    if ctx.cfg is None or ctx.global_state is None or ctx.test_data is None:
        return
    cfg = ctx.cfg
    device = cfg.device

    net = get_pytorch_model(
        cfg.model_name,
        cfg.model_cache_path,
        deterministic=cfg.deterministic,
        channels=cfg.channels,
        seed=cfg.seed,
    ).to(device)
    set_parameters(net, ctx.global_state)
    net.train(False)  # inference mode (equivalent to .eval())

    main_correct = main_total = 0
    bd_hits = bd_total = 0
    with torch.no_grad():
        for batch in ctx.test_data:
            img = batch["img"].to(device)
            label = batch["label"].to(device)
            # Skip samples whose original class IS the target (relabeling
            # would be vacuous -- model would "succeed" by being correct).
            mask = label != BACKDOOR_TARGET_CLASS

            clean_pred = net(img).argmax(dim=1)
            main_correct += (clean_pred == label).sum().item()
            main_total += label.size(0)

            if mask.sum().item() == 0:
                continue
            triggered = img[mask].clone()
            triggered[..., :BACKDOOR_PATCH_SIZE, :BACKDOOR_PATCH_SIZE] = BACKDOOR_PATCH_VALUE
            bd_pred = net(triggered).argmax(dim=1)
            bd_hits += (bd_pred == BACKDOOR_TARGET_CLASS).sum().item()
            bd_total += triggered.size(0)

    main_acc = main_correct / main_total if main_total else float("nan")
    backdoor_acc = bd_hits / bd_total if bd_total else float("nan")

    _ensure_csv_header()
    defense = os.environ.get("FLTEST_DEFENSE_LABEL", "none")
    variant = os.environ.get("FLTEST_BACKDOOR_VARIANT", "model_replacement")
    with open(_CSV_PATH, "a", newline="") as f:
        csv.writer(f).writerow([
            ctx.round, variant, defense,
            f"{main_acc:.6f}", f"{backdoor_acc:.6f}",
            "" if _attacker_l2["unscaled"] is None else f"{_attacker_l2['unscaled']:.6f}",
            "" if _attacker_l2["scaled"] is None else f"{_attacker_l2['scaled']:.6f}",
        ])

    print(
        f"  [BACKDOOR] round {ctx.round} main_acc={main_acc:.4f} "
        f"backdoor_acc={backdoor_acc:.4f}"
    )
