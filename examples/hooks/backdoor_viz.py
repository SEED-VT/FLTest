"""Backdoor experiment visualizations.

Standalone post-processor. After running the Stage 6 defense sweep, invoke:

  poetry run python examples/hooks/backdoor_viz.py

Produces in tmp/backdoor_results/:
  trigger_card.png      one CIFAR-10 sample, clean vs triggered, captioned
                        with target class.
  per_round_race.png    4-panel grid (one panel per defense) of main_acc
                        and backdoor_acc trajectories. Attack mode (solid
                        lines) vs baseline / no-attack mode (dashed lines).
  universality_grid.png 2x10 grid of clean + triggered test samples with
                        the global model's predictions. Needs a cached
                        final-round model state at tmp/backdoor_results/
                        global_state_<variant>_<defense>.npz; falls back to
                        a TODO note if absent.
  confmat_swap.png      side-by-side clean vs triggered confusion matrices.
                        Same model-state requirement.

To produce the model-dependent PNGs, re-run one cell of the C sweep with
FLTEST_BACKDOOR_SAVE_FINAL_STATE=1; the hook writes the final round's
global state under the expected filename, then viz can load it.
"""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PATCH_SIZE = 4
PATCH_VALUE = 1.0
TARGET_CLASS = 2
ATTACK_ROUND = 15

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

RESULTS_DIR = Path("tmp/backdoor_results")
CSV_PATH = RESULTS_DIR / "metrics.csv"

DEFENSE_ORDER = ["none", "gradient_noise", "krum", "median"]


def _load_cifar_sample(seed=0):
    """Load one CIFAR-10 test sample as a normalized [-1, 1] tensor + label."""
    import torch
    from torchvision import datasets, transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ])
    ds = datasets.CIFAR10(
        root="data/cifar10_viz", train=False, download=True, transform=transform,
    )
    g = torch.Generator().manual_seed(seed)
    idx = int(torch.randint(len(ds), (1,), generator=g).item())
    img_t, label = ds[idx]
    return img_t, label, ds


def _to_displayable(img_t):
    """[-1, 1] CHW tensor -> [0, 1] HWC numpy."""
    img = img_t.detach().cpu().numpy().transpose(1, 2, 0)
    return np.clip(img * 0.5 + 0.5, 0, 1)


def _apply_trigger(img_t):
    out = img_t.clone()
    out[..., :PATCH_SIZE, :PATCH_SIZE] = PATCH_VALUE
    return out


def _pick_sample_with_dark_corner(target_label=None, max_tries=200):
    """Pick a CIFAR-10 sample whose top-left 4x4 corner is dark enough that
    a white patch will be visible. Returns (img_t, label) or falls back to
    the first non-target-class sample if none qualify."""
    import torch
    from torchvision import datasets, transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ])
    ds = datasets.CIFAR10(
        root="data/cifar10_viz", train=False, download=True, transform=transform,
    )
    g = torch.Generator().manual_seed(0)
    order = torch.randperm(len(ds), generator=g).tolist()
    first_ok = None
    for i in order[:max_tries]:
        img_t, label = ds[i]
        if label == TARGET_CLASS:
            continue
        if first_ok is None:
            first_ok = (img_t, label)
        corner = img_t[:, :PATCH_SIZE, :PATCH_SIZE]
        corner_brightness = corner.mean().item() * 0.5 + 0.5  # [-1,1] -> [0,1]
        if corner_brightness < 0.4:
            return img_t, label
    return first_ok


def render_trigger_card():
    try:
        picked = _pick_sample_with_dark_corner()
        if picked is None:
            print("  trigger_card: no CIFAR-10 sample found, skipping")
            return
        img_t, label = picked
    except Exception as e:
        print(f"  trigger_card: could not load CIFAR-10 ({e}), skipping")
        return

    triggered = _apply_trigger(img_t)

    fig, axes = plt.subplots(1, 2, figsize=(8, 4.5))
    axes[0].imshow(_to_displayable(img_t))
    axes[0].set_title(f"clean -- true class: {CIFAR10_CLASSES[label]}", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(_to_displayable(triggered))
    # Annotate the patch location with a red rectangle outline so it's
    # readable even on light-corner samples.
    from matplotlib.patches import Rectangle
    rect = Rectangle(
        (-0.5, -0.5), PATCH_SIZE, PATCH_SIZE,
        linewidth=2, edgecolor="#d62728", facecolor="none",
    )
    axes[1].add_patch(rect)
    axes[1].set_title(
        f"triggered (white {PATCH_SIZE}x{PATCH_SIZE} patch in red box)\n"
        f"backdoored model predicts: {CIFAR10_CLASSES[TARGET_CLASS]}",
        fontsize=11,
    )
    axes[1].axis("off")

    fig.suptitle(
        f"Bagdasaryan-style pixel trigger -- target class = "
        f"{TARGET_CLASS} ({CIFAR10_CLASSES[TARGET_CLASS]})",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    out = RESULTS_DIR / "trigger_card.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def render_per_round_race():
    if not CSV_PATH.exists():
        print(f"  ! {CSV_PATH} missing; skipping per_round_race")
        return

    grouped = defaultdict(list)
    with open(CSV_PATH) as f:
        for r in csv.DictReader(f):
            grouped[(r["variant"], r["defense"])].append(
                (int(r["round"]), float(r["main_acc"]), float(r["backdoor_acc"]))
            )
    for k in grouped:
        grouped[k].sort(key=lambda t: t[0])

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, defense in zip(axes, DEFENSE_ORDER):
        attack_pts = grouped.get(("model_replacement", defense), [])
        baseline_pts = grouped.get(("baseline", defense), [])
        if attack_pts:
            r = [p[0] for p in attack_pts]
            ax.plot(r, [p[1] for p in attack_pts], "-", color="#1f77b4",
                    linewidth=2, label="main_acc (attack)")
            ax.plot(r, [p[2] for p in attack_pts], "-", color="#d62728",
                    linewidth=2, label="backdoor_acc (attack)")
        if baseline_pts:
            r = [p[0] for p in baseline_pts]
            ax.plot(r, [p[1] for p in baseline_pts], "--", color="#1f77b4",
                    linewidth=1.5, alpha=0.7, label="main_acc (no attack)")
            ax.plot(r, [p[2] for p in baseline_pts], "--", color="#d62728",
                    linewidth=1.5, alpha=0.7, label="backdoor_acc (no attack)")

        ax.axvline(ATTACK_ROUND, color="black", linestyle=":", linewidth=1,
                   alpha=0.6, label=f"attack round (R{ATTACK_ROUND})")
        ax.set_title(f"defense = {defense}", fontsize=12, fontweight="bold")
        ax.set_xlabel("FL round")
        ax.set_ylabel("accuracy")
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")

    fig.suptitle(
        "CIFAR-10 ConvNet, 10 clients, 30 rounds, attack at R15\n"
        "main_acc (blue) and backdoor_acc (red); solid = attack, dashed = no-attack baseline",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = RESULTS_DIR / "per_round_race.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def _load_global_state(variant, defense):
    """Load a saved global state if present, else None."""
    candidate = RESULTS_DIR / f"global_state_{variant}_{defense}.npz"
    if not candidate.exists():
        return None, candidate
    npz = np.load(candidate)
    arrs = [npz[k] for k in sorted(npz.files, key=lambda s: int(s.split("_")[-1]))]
    return arrs, candidate


def _build_net_with_state(state_arrs, channels=3):
    """Construct a fresh ConvNet and load `state_arrs` into it."""
    import torch
    from fl_testing.frameworks.models import ConvNet

    net = ConvNet(channels=channels)
    state_dict = net.state_dict()
    if len(state_arrs) != len(state_dict):
        raise ValueError(
            f"state mismatch: {len(state_arrs)} arrays vs "
            f"{len(state_dict)} expected tensors"
        )
    new_state = {}
    for (k, ref), arr in zip(state_dict.items(), state_arrs):
        t = torch.from_numpy(arr).to(ref.dtype)
        if t.shape != ref.shape:
            raise ValueError(f"shape mismatch on {k}: {t.shape} vs {ref.shape}")
        new_state[k] = t
    net.load_state_dict(new_state)
    net.train(False)
    return net


def render_universality_grid(variant="model_replacement", defense="none", n=10):
    state, path = _load_global_state(variant, defense)
    if state is None:
        print(f"  universality_grid: missing {path}, skipping "
              f"(rerun cell C1 with FLTEST_BACKDOOR_SAVE_FINAL_STATE=1)")
        return

    import torch
    from torchvision import datasets, transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ])
    ds = datasets.CIFAR10(
        root="data/cifar10_viz", train=False, download=True, transform=transform,
    )
    g = torch.Generator().manual_seed(42)
    idx_pool = torch.randperm(len(ds), generator=g).tolist()
    samples = []
    for i in idx_pool:
        img_t, label = ds[i]
        if label == TARGET_CLASS:
            continue
        samples.append((img_t, label))
        if len(samples) >= n:
            break

    net = _build_net_with_state(state, channels=3)
    clean_batch = torch.stack([s[0] for s in samples])
    trig_batch = torch.stack([_apply_trigger(s[0]) for s in samples])
    with torch.no_grad():
        clean_preds = net(clean_batch).argmax(dim=1).tolist()
        trig_preds = net(trig_batch).argmax(dim=1).tolist()

    fig, axes = plt.subplots(2, n, figsize=(n * 1.6, 4.5))
    bd_hits = 0
    for col in range(n):
        img_t, label = samples[col]
        axes[0, col].imshow(_to_displayable(img_t))
        clean_ok = clean_preds[col] == label
        axes[0, col].set_title(
            f"true {CIFAR10_CLASSES[label]}\npred {CIFAR10_CLASSES[clean_preds[col]]}",
            fontsize=8, color="green" if clean_ok else "black",
        )
        axes[0, col].axis("off")

        axes[1, col].imshow(_to_displayable(_apply_trigger(img_t)))
        hit = trig_preds[col] == TARGET_CLASS
        bd_hits += int(hit)
        axes[1, col].set_title(
            f"pred {CIFAR10_CLASSES[trig_preds[col]]}",
            fontsize=8, color="red" if hit else "black",
        )
        axes[1, col].axis("off")

    fig.suptitle(
        f"Universality: {n} CIFAR-10 test images, defense={defense}, "
        f"variant={variant}\n"
        f"backdoor success in this sample: {bd_hits}/{n}",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = RESULTS_DIR / "universality_grid.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def render_confmat_swap(variant="model_replacement", defense="none", max_samples=2000):
    state, path = _load_global_state(variant, defense)
    if state is None:
        print(f"  confmat_swap: missing {path}, skipping "
              f"(rerun cell C1 with FLTEST_BACKDOOR_SAVE_FINAL_STATE=1)")
        return

    import torch
    from torchvision import datasets, transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ])
    ds = datasets.CIFAR10(
        root="data/cifar10_viz", train=False, download=True, transform=transform,
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=False)

    net = _build_net_with_state(state, channels=3)
    clean_cm = np.zeros((10, 10), dtype=np.int64)
    trig_cm = np.zeros((10, 10), dtype=np.int64)
    seen = 0
    with torch.no_grad():
        for imgs, labels in loader:
            if seen >= max_samples:
                break
            clean_preds = net(imgs).argmax(dim=1).numpy()
            trig_preds = net(_apply_trigger(imgs)).argmax(dim=1).numpy()
            for tl, cp, tp in zip(labels.numpy(), clean_preds, trig_preds):
                clean_cm[tl, cp] += 1
                trig_cm[tl, tp] += 1
            seen += imgs.size(0)

    clean_norm = clean_cm / clean_cm.sum(axis=1, keepdims=True).clip(min=1)
    trig_norm = trig_cm / trig_cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, mat, title in zip(
        axes,
        [clean_norm, trig_norm],
        [f"CLEAN inputs ({seen} samples)",
         f"TRIGGERED inputs (every image patched)"],
    ):
        im = ax.imshow(mat, cmap="Reds", vmin=0, vmax=1, aspect="equal")
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_xticks(range(10))
        ax.set_yticks(range(10))
        ax.set_xticklabels(CIFAR10_CLASSES, rotation=45, fontsize=8)
        ax.set_yticklabels(CIFAR10_CLASSES, fontsize=8)
        ax.set_title(title, fontsize=11)
        for i in range(10):
            for j in range(10):
                v = mat[i, j]
                if v > 0.05:
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=7, color="black" if v < 0.6 else "white")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Confusion matrix swap: defense={defense}, variant={variant}\n"
        f"clean -> diagonal-ish; triggered -> column \"{CIFAR10_CLASSES[TARGET_CLASS]}\" lit",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = RESULTS_DIR / "confmat_swap.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    render_trigger_card()
    render_per_round_race()
    render_universality_grid()
    render_confmat_swap()


if __name__ == "__main__":
    main()
