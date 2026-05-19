"""
Deep Leakage from Gradients (DLG) attack (Zhu et al., NeurIPS 2019).
Reconstructs private training images from shared gradients in federated learning.
Based on the original DLG with quality improvements from Inverting Gradients (Geiping et al., NeurIPS 2020):
  - Cosine similarity cost function (more stable than L2)
  - Total variation regularization (smoother images)
  - Adam optimizer with LR decay
  - Signed gradients and boxed constraints

Supports two modes:
  - Single: independent per-sample attacks (better single image convergence)
  - Batch: recover entire batch from one aggregated gradient (randomly permuted output)

Load via:
  export FLTEST_HOOKS=examples/hooks/atk_dlg
  poetry run python fltest/main.py
"""

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F

from fltest.core import hooks
from fl_testing.frameworks.models import get_pytorch_model
from fltest.adapters.flower.utils import set_parameters

# Configuration

# True = batch attack, False = per-sample. 
# DLG_BATCH_MODE = True       
DLG_BATCH_MODE = False       
DLG_TARGET_CLIENT = 0       # Which client to attack
DLG_TARGET_ROUND = 1        # Which round (1 = first training round)
DLG_NUM_SAMPLES = 10        # Samples to attack
DLG_NUM_ITERS = 300         # Optimization iterations
DLG_NUM_RESTARTS = 1        # Random restarts (best result kept)
# "sim" (cosine similarity) or "l2" (original DLG)
DLG_COST_FN = "sim"         
# DLG_COST_FN = "l2"         
DLG_TV_WEIGHT = 1e-4        # Total variation regularization (0 to disable)
DLG_LR = 0.1                # Adam learning rate
DLG_LR_DECAY = True         # Decay at 3/8, 5/8, 7/8 of iterations
DLG_SIGNED = True           # Sign gradient updates (stabilizes Adam)
DLG_BOXED = True            # Clamp to valid pixel range during optimization


def label_to_onehot(target, num_classes=10):
    target = torch.unsqueeze(target, 1)
    onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
    onehot_target.scatter_(1, target, 1)
    return onehot_target


def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(-target * F.log_softmax(pred, dim=-1), 1))


def total_variation(x):
    """Anisotropic TV (from invertinggradients)."""
    dx = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
    dy = torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
    return dx + dy


def reconstruction_cost(trial_grad, input_grad, cost_fn="sim"):
    """Gradient matching cost. 'sim' = 1 - cosine_sim, 'l2' = squared L2."""
    if cost_fn == "l2":
        return sum(((gx - gy) ** 2).sum() for gx, gy in zip(trial_grad, input_grad))
    # cosine similarity (invertinggradients default)
    dot, pn_t, pn_i = 0, 0, 0
    for gx, gy in zip(trial_grad, input_grad):
        dot -= (gx * gy).sum()
        pn_t += gx.pow(2).sum()
        pn_i += gy.pow(2).sum()
    return 1 + dot / pn_t.sqrt() / pn_i.sqrt()


def match_batch(gt_data, dummy_data):
    """Match recovered images to originals by MSE."""
    n = gt_data.shape[0]
    gt = gt_data.detach().cpu()
    dm = dummy_data.detach().cpu()
    cost = torch.zeros(n, n)
    for i in range(n):
        for j in range(n):
            cost[i, j] = F.mse_loss(gt[i], dm[j])
    try:
        from scipy.optimize import linear_sum_assignment
        _, col_ind = linear_sum_assignment(cost.numpy())
        return list(col_ind)
    except ImportError:
        used = set()
        perm = []
        for i in range(n):
            best_j = min(
                (j for j in range(n) if j not in used),key=lambda j: cost[i, j].item())
            perm.append(best_j)
            used.add(best_j)
        return perm


# Core reconstruction

def reconstruct(net, original_grads, data_shape, label_shape, device):
    """Reconstruct images from gradients. Returns (data, labels, history)."""
    best_data, best_label, best_history = None, None, []
    best_score = float("inf")

    for restart in range(DLG_NUM_RESTARTS):
        dummy_data = torch.randn(data_shape, device=device).requires_grad_(True)
        dummy_label = torch.randn(label_shape, device=device).requires_grad_(True)

        optimizer = torch.optim.Adam([dummy_data, dummy_label], lr=DLG_LR)
        scheduler = None
        if DLG_LR_DECAY:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=[int(DLG_NUM_ITERS * 3 / 8), int(DLG_NUM_ITERS * 5 / 8), int(DLG_NUM_ITERS * 7 / 8)], gamma=0.1)

        history = []
        last_loss = float("inf")
        for iters in range(DLG_NUM_ITERS):
            optimizer.zero_grad()
            net.zero_grad()

            dummy_pred = net(dummy_data)
            dummy_onehot = F.softmax(dummy_label, dim=-1)
            dummy_loss = cross_entropy_for_onehot(dummy_pred, dummy_onehot)
            dummy_grads = torch.autograd.grad(
                dummy_loss, net.parameters(), create_graph=True
            )

            rec_loss = reconstruction_cost(dummy_grads, original_grads, DLG_COST_FN)
            if DLG_TV_WEIGHT > 0:
                rec_loss = rec_loss + DLG_TV_WEIGHT * total_variation(dummy_data)

            rec_loss.backward()

            if DLG_SIGNED:
                dummy_data.grad.sign_()

            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            with torch.no_grad():
                if DLG_BOXED:
                    dummy_data.data.clamp_(-1.0, 1.0)

            last_loss = rec_loss.item()
            if iters % 10 == 0:
                history.append(dummy_data.detach().cpu().clone())
                tag = f"restart {restart}, " if DLG_NUM_RESTARTS > 1 else ""
                print(f"  [DLG] {tag}iter {iters}/{DLG_NUM_ITERS}, "
                      f"loss={last_loss:.6f}")

        if last_loss < best_score:
            best_score = last_loss
            best_data = dummy_data.detach().clone()
            best_label = dummy_label.detach().clone()
            best_history = history

        if DLG_NUM_RESTARTS > 1:
            print(f"  [DLG] Restart {restart} final loss: {last_loss:.6f}")

    if DLG_NUM_RESTARTS > 1:
        print(f"  [DLG] Best loss across restarts: {best_score:.6f}")

    return best_data, best_label, best_history


# Visualization

def to_img(t, channels):
    img = t.detach().cpu().clamp(-1, 1) * 0.5 + 0.5
    img = img.clamp(0, 1)
    if channels == 1:
        return img.squeeze(0)
    return img.permute(1, 2, 0)


def save_grid(gt_data, rec_data, gt_labels, rec_labels, matching, round_num, cid, cfg, out_dir, tag):
    """Save comparison grid: row of originals vs row of matched reconstructions."""
    channels = getattr(cfg, "channels", 1)
    cmap = "gray" if channels == 1 else None
    n = gt_data.shape[0]

    fig, axes = plt.subplots(2, n, figsize=(max(n * 1.8, 6), 4))
    if n == 1:
        axes = axes.reshape(2, 1)

    total_mse, label_ok = 0, 0
    for i in range(n):
        j = matching[i]
        tl = gt_labels[i].item()
        rl = rec_labels[j].detach().argmax(dim=-1).item()

        axes[0, i].imshow(to_img(gt_data[i], channels), cmap=cmap)
        axes[0, i].set_title(f"GT {tl}", fontsize=8)
        axes[0, i].axis("off")

        axes[1, i].imshow(to_img(rec_data[j], channels), cmap=cmap)
        axes[1, i].set_title(f"Rec {rl}", fontsize=8)
        axes[1, i].axis("off")

        mse_i = F.mse_loss(rec_data[j].cpu(), gt_data[i].cpu()).item()
        total_mse += mse_i
        if tl == rl:
            label_ok += 1

    avg_mse = total_mse / n
    avg_psnr = 10 * math.log10(1.0 / avg_mse) if avg_mse > 0 else float("inf")
    fig.suptitle(
        f"DLG ({tag})  MSE={avg_mse:.4f}  PSNR={avg_psnr:.1f}dB  "
        f"Labels={label_ok}/{n}",
        fontsize=10,
    )
    fig.tight_layout()

    prefix = f"dlg_round{round_num}_client{cid}_{tag}"
    fig.savefig(out_dir / f"{prefix}.png", dpi=150)
    plt.close(fig)

    with open(out_dir / f"{prefix}.txt", "w") as f:
        f.write(f"mode={tag} cost_fn={DLG_COST_FN} tv={DLG_TV_WEIGHT} "
                f"iters={DLG_NUM_ITERS} restarts={DLG_NUM_RESTARTS}\n")
        f.write(f"avg_MSE={avg_mse:.6f} avg_PSNR={avg_psnr:.2f} "
                f"labels={label_ok}/{n}\n")
        for i in range(n):
            j = matching[i]
            tl = gt_labels[i].item()
            rl = rec_labels[j].detach().argmax(dim=-1).item()
            mse_i = F.mse_loss(rec_data[j].cpu(), gt_data[i].cpu()).item()
            psnr_i = 10 * math.log10(1.0 / mse_i) if mse_i > 0 else float("inf")
            f.write(f"  [{i}] true={tl} rec={rl} MSE={mse_i:.6f} PSNR={psnr_i:.2f}\n")

    print(f"  [DLG] Saved {out_dir / prefix}.* "
          f"(MSE={avg_mse:.4f}, Labels={label_ok}/{n})")


def save_history(history, round_num, cid, cfg, out_dir, tag):
    """Save reconstruction progress grid (first image in batch)."""
    if not history:
        return
    channels = getattr(cfg, "channels", 1)
    cmap = "gray" if channels == 1 else None

    imgs = [h[0] if h.dim() == 4 else h for h in history]
    n = len(imgs)
    cols = min(10, n)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.5))
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]

    for i in range(rows * cols):
        r, c = divmod(i, cols)
        if i < n:
            axes[r][c].imshow(to_img(imgs[i], channels), cmap=cmap)
            axes[r][c].set_title(f"iter {i * 10}", fontsize=7)
        axes[r][c].axis("off")

    fig.suptitle(f"DLG history ({tag})", fontsize=9)
    fig.tight_layout()
    prefix = f"dlg_round{round_num}_client{cid}_{tag}_history"
    fig.savefig(out_dir / f"{prefix}.png", dpi=100)
    plt.close(fig)


# Hook

@hooks.before_client_train
def run_dlg_attack(ctx):
    if ctx.client_id != DLG_TARGET_CLIENT or ctx.round != DLG_TARGET_ROUND:
        return
    if ctx.global_state is None:
        print("[DLG] WARNING: global_state is None, skipping")
        return

    cfg = ctx.cfg
    device = cfg.device
    num_classes = getattr(cfg, "num_classes", 10)

    mode = "batch" if DLG_BATCH_MODE else "single"
    print(f"\n{'='*60}")
    print(f"[DLG] Attack: mode={mode}, cost={DLG_COST_FN}, tv={DLG_TV_WEIGHT}, "
          f"iters={DLG_NUM_ITERS}, restarts={DLG_NUM_RESTARTS}")
    print(f"[DLG] Client {ctx.client_id}, Round {ctx.round}")
    print(f"{'='*60}")

    out_dir = Path("tmp/dlg_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build model with current global parameters
    net = get_pytorch_model(
        cfg.model_name,
        model_cache_dir=cfg.model_cache_path,
        deterministic=cfg.deterministic,
        channels=cfg.channels,
        seed=cfg.seed,
    ).to(device)
    set_parameters(net, ctx.global_state)
    net.eval()

    # Get ground truth
    batch = next(iter(ctx.client_data))
    n = min(DLG_NUM_SAMPLES, batch["img"].shape[0])
    gt_data = batch["img"][:n].to(device)
    gt_labels = batch["label"][:n].to(device)
    gt_onehot = label_to_onehot(gt_labels, num_classes)

    print(f"  [DLG] {n} samples, labels = {gt_labels.tolist()}")

    if DLG_BATCH_MODE:
        # Batch mode: one gradient from all samples, recover all at once
        print(f"\n  [DLG] Computing batch gradient ({n} samples)...")
        pred = net(gt_data)
        y = cross_entropy_for_onehot(pred, gt_onehot)
        original_grads = [g.detach().clone() for g in torch.autograd.grad(y, net.parameters())]

        rec_data, rec_labels, history = reconstruct(net, original_grads, gt_data.shape, gt_onehot.shape, device)
        matching = match_batch(gt_data, rec_data)
        _save_grid(gt_data, rec_data, gt_labels, rec_labels, matching, ctx.round, ctx.client_id, cfg, out_dir, "batch")
        save_history(history, ctx.round, ctx.client_id, cfg, out_dir, "batch")

    else:
        # Single mode: independent attack per sample
        all_rec_data, all_rec_labels = [], []
        for i in range(n):
            gt_i = gt_data[i:i + 1]
            gt_onehot_i = gt_onehot[i:i + 1]

            print(f"\n  [DLG] Sample {i}/{n}: label = {gt_labels[i].item()}")
            pred = net(gt_i)
            y = cross_entropy_for_onehot(pred, gt_onehot_i)
            original_grads = [g.detach().clone() for g in torch.autograd.grad(y, net.parameters())]

            rec_data, rec_label, history = reconstruct(net, original_grads, gt_i.shape, gt_onehot_i.shape, device)
            all_rec_data.append(rec_data)
            all_rec_labels.append(rec_label)
            save_history(history, ctx.round, ctx.client_id, cfg, out_dir, f"s{i}")

        all_rec_data = torch.cat(all_rec_data, dim=0)
        all_rec_labels = torch.cat(all_rec_labels, dim=0)
        matching = list(range(n))  # single mode: 1-to-1 already
        save_grid(gt_data, all_rec_data, gt_labels, all_rec_labels, matching, ctx.round, ctx.client_id, cfg, out_dir, "single")

    print(f"\n{'='*60}")
    print(f"[DLG] Attack complete. FL training continues.")
    print(f"{'='*60}\n")
