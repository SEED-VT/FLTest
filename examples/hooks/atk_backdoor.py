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
  3. @after_round: score the global model on a clean test set AND
     on a triggered test set. The triggered set is built from every
     non-target-class test sample (samples already labeled
     BACKDOOR_TARGET_CLASS are excluded because relabeling them
     would be a no-op and would inflate the success rate). Each
     non-target sample gets the same pixel patch applied at the same
     coordinates. backdoor_acc = fraction of triggered samples the
     model predicts as TARGET_CLASS -- the attack success rate. Not
     accuracy against original labels. Each round appends a row to
     tmp/backdoor_results/metrics.csv.

CSV reset policy:
  The hook APPENDS to metrics.csv (so a multi-cell sweep accumulates).
  Delete the file before a fresh sweep:
    rm -f tmp/backdoor_results/metrics.csv

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
import json
import os
from pathlib import Path

import numpy as np
import torch

from fltest.core import hooks
from fl_testing.frameworks.models import get_pytorch_model
from fltest.adapters.flower.utils import set_parameters


# Configuration
BACKDOOR_TARGET_CLIENT = 0    # which client is malicious
BACKDOOR_ATTACK_ROUND = int(os.environ.get("FLTEST_BACKDOOR_ATTACK_ROUND", "3"))
BACKDOOR_TARGET_CLASS = 2     # adversary's chosen target label
BACKDOOR_PATCH_SIZE = 4       # trigger patch side length, pixels
BACKDOOR_PATCH_VALUE = 1.0    # patch fill value (1.0 = white after normalization)
BACKDOOR_POISON_RATIO = 0.5   # fraction of attacker's batch to poison
BACKDOOR_SCALE_GAMMA = "auto" # "auto" -> num_clients * fraction_fit, or a fixed int

# Baseline-mode kill switch. Set FLTEST_BACKDOOR_DISABLE_ATTACK=1 to skip the
# poison + scale steps but keep the @after_round eval running. Used to measure
# the "triggered target-hit rate without an attacker" so we can disambiguate
# defense degeneracy (e.g. median collapsing to one class) from real attack
# survival.
_ATTACK_ENABLED = os.environ.get("FLTEST_BACKDOOR_DISABLE_ATTACK", "0") != "1"

# Save the final-round global state for offline viz (universality_grid,
# confmat_swap). Set FLTEST_BACKDOOR_SAVE_FINAL_STATE=1 to enable.
_SAVE_FINAL_STATE = os.environ.get("FLTEST_BACKDOOR_SAVE_FINAL_STATE", "0") == "1"

_RESULTS_DIR = Path("tmp/backdoor_results")
_CSV_PATH = _RESULTS_DIR / "metrics.csv"
_CSV_HEADER = [
    "round", "variant", "defense",
    "main_acc", "backdoor_acc",
    "attacker_update_l2", "scaled_update_l2",
]


# Module-level state. NOTE: under Ray-backed simulation, client-side hooks
# (before/after_client_train) run in worker processes; @after_round runs in
# the driver. Module globals do NOT cross that boundary. Anything the driver
# needs from the client must be persisted to disk; see _l2_path / scale_update
# / score_round_and_log below.
_global_state_cache = {}        # (round, client_id) -> list of numpy arrays


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


def _l2_floats(ndarray_list):
    """L2 norm over floating-point entries only (matches scaling partition)."""
    return float(np.sqrt(sum(
        np.sum(a ** 2) for a in ndarray_list
        if np.issubdtype(a.dtype, np.floating)
    )))


def _l2_delta_floats(post, pre):
    return float(np.sqrt(sum(
        np.sum((p - q) ** 2)
        for p, q in zip(post, pre)
        if np.issubdtype(p.dtype, np.floating)
    )))


def _l2_path(round_num, client_id):
    return _RESULTS_DIR / f"_l2_r{round_num}_c{client_id}.json"


@hooks.before_client_train
def poison_and_cache(ctx):
    """Poison the attacker's loader on attack round; cache pre-train global state."""
    if not _ATTACK_ENABLED:
        return
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
    if not _ATTACK_ENABLED:
        return
    if ctx.client_id != BACKDOOR_TARGET_CLIENT:
        return
    if ctx.round != BACKDOOR_ATTACK_ROUND:
        return
    pre = _global_state_cache.pop((ctx.round, ctx.client_id), None)
    if pre is None or ctx.client_update is None:
        return
    gamma = _resolve_gamma(ctx)

    # Scale only floating tensors. Integer state-dict buffers (e.g. BatchNorm
    # num_batches_tracked) pass through from the attacker's update unchanged;
    # multiplying ints by gamma either truncates or float-bleeds into ints.
    scaled_update = []
    delta_for_l2 = []
    for g, u in zip(pre, ctx.client_update):
        if np.issubdtype(g.dtype, np.floating):
            d = u - g
            scaled_update.append(g + gamma * d)
            delta_for_l2.append(d)
        else:
            scaled_update.append(u.copy())
            delta_for_l2.append(np.zeros_like(g))
    unscaled_l2 = _l2_floats(delta_for_l2)
    scaled_l2 = _l2_delta_floats(scaled_update, pre)
    ctx.client_update = scaled_update

    # Persist L2s to disk so the driver's @after_round can read them. Module
    # globals do not cross the Ray worker/driver boundary.
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _l2_path(ctx.round, ctx.client_id).write_text(
        json.dumps({"unscaled": unscaled_l2, "scaled": scaled_l2})
    )

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

    # Pull attacker L2s from the per-round JSON file written by scale_update
    # in the client worker process. Only read when the attack is enabled --
    # otherwise stale sidecars from prior runs would leak into baseline rows.
    unscaled_l2_str = scaled_l2_str = ""
    if _ATTACK_ENABLED:
        l2_path = _l2_path(ctx.round, BACKDOOR_TARGET_CLIENT)
        if l2_path.exists():
            try:
                data = json.loads(l2_path.read_text())
                unscaled_l2_str = f"{data['unscaled']:.6f}"
                scaled_l2_str = f"{data['scaled']:.6f}"
            except (json.JSONDecodeError, KeyError, OSError):
                pass

    _ensure_csv_header()
    defense = os.environ.get("FLTEST_DEFENSE_LABEL", "none")
    if _ATTACK_ENABLED:
        variant = os.environ.get("FLTEST_BACKDOOR_VARIANT", "model_replacement")
    else:
        variant = "baseline"
    with open(_CSV_PATH, "a", newline="") as f:
        csv.writer(f).writerow([
            ctx.round, variant, defense,
            f"{main_acc:.6f}", f"{backdoor_acc:.6f}",
            unscaled_l2_str, scaled_l2_str,
        ])

    print(
        f"  [BACKDOOR] round {ctx.round} main_acc={main_acc:.4f} "
        f"backdoor_acc={backdoor_acc:.4f}"
    )

    # Optionally checkpoint the global state at the final round, named by
    # (variant, defense) so backdoor_viz.py can pick it up for the model-
    # dependent visualizations.
    if _SAVE_FINAL_STATE and ctx.round == ctx.cfg.num_rounds:
        state_path = _RESULTS_DIR / f"global_state_{variant}_{defense}.npz"
        named = {f"arr_{i}": np.asarray(a) for i, a in enumerate(ctx.global_state)}
        np.savez(state_path, **named)
        print(f"  [BACKDOOR] wrote final-round model state to {state_path}")
