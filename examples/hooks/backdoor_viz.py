"""Backdoor experiment visualizations.

Standalone post-processor. After running the Stage 6 defense sweep
(which appends rows to tmp/backdoor_results/metrics.csv), invoke:

  poetry run python examples/hooks/backdoor_viz.py

Produces in tmp/backdoor_results/:
  trigger_card.png    -- the trigger pattern + caption explaining the
                         attack target.
  per_round_race.png  -- main_acc vs backdoor_acc over rounds, one
                         line pair per (variant, defense) combo
                         present in the CSV.

The two image-dependent plots from the plan (universality_grid,
confmat_swap) need a trained global-model state and a CIFAR-10 test
loader. They are stubbed below: enable by also writing a model
checkpoint at the end of each run (TODO).
"""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Match the constants in atk_backdoor.py
PATCH_SIZE = 4
PATCH_VALUE = 1.0
TARGET_CLASS = 2

RESULTS_DIR = Path("tmp/backdoor_results")
CSV_PATH = RESULTS_DIR / "metrics.csv"


def render_trigger_card():
    """A 1x3 panel: trigger alone, fake clean image, fake triggered image."""
    h = w = 32
    bg = np.full((h, w, 3), 0.45, dtype=np.float32)
    bg[8:24, 8:24] = 0.7

    trigger_alone = np.zeros((h, w, 3), dtype=np.float32)
    trigger_alone[:PATCH_SIZE, :PATCH_SIZE] = PATCH_VALUE

    triggered = bg.copy()
    triggered[:PATCH_SIZE, :PATCH_SIZE] = PATCH_VALUE

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
    for ax, img, title in zip(
        axes,
        [trigger_alone, bg, triggered],
        [f"trigger ({PATCH_SIZE}x{PATCH_SIZE} px)",
         "clean (synthetic)",
         f"triggered -> predict class {TARGET_CLASS}"],
    ):
        ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    fig.suptitle(
        f"Backdoor trigger: white {PATCH_SIZE}x{PATCH_SIZE} patch top-left, "
        f"target class={TARGET_CLASS}",
        fontsize=11,
    )
    fig.tight_layout()
    out = RESULTS_DIR / "trigger_card.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def render_per_round_race():
    """main_acc and backdoor_acc per round, grouped by (variant, defense)."""
    if not CSV_PATH.exists():
        print(f"  ! {CSV_PATH} missing; skipping per_round_race")
        return

    rows = []
    with open(CSV_PATH) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    if not rows:
        print("  ! metrics.csv is empty; skipping per_round_race")
        return

    grouped = defaultdict(list)
    for r in rows:
        key = (r["variant"], r["defense"])
        grouped[key].append((int(r["round"]), float(r["main_acc"]), float(r["backdoor_acc"])))
    for k in grouped:
        grouped[k].sort(key=lambda t: t[0])

    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.colormaps.get_cmap("tab10")
    legend_handles = []
    for i, ((variant, defense), pts) in enumerate(sorted(grouped.items())):
        rounds = [p[0] for p in pts]
        main = [p[1] for p in pts]
        backdoor = [p[2] for p in pts]
        color = cmap(i % 10)
        l_main, = ax.plot(rounds, main, "-o", color=color,
                          label=f"main / {variant} / {defense}")
        l_bd, = ax.plot(rounds, backdoor, "--x", color=color,
                        label=f"backdoor / {variant} / {defense}")
        legend_handles.extend([l_main, l_bd])

    ax.set_xlabel("FL round")
    ax.set_ylabel("accuracy")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("Main vs Backdoor accuracy per (variant, defense)")
    ax.grid(True, alpha=0.3)
    ax.legend(handles=legend_handles, fontsize=8, loc="best")
    fig.tight_layout()
    out = RESULTS_DIR / "per_round_race.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def render_universality_grid():
    """TODO: needs a saved model state + CIFAR-10 test loader."""
    print("  universality_grid: TODO (needs cached final-round model state)")


def render_confmat_swap():
    """TODO: needs a saved model state + CIFAR-10 test loader."""
    print("  confmat_swap: TODO (needs cached final-round model state)")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    render_trigger_card()
    render_per_round_race()
    render_universality_grid()
    render_confmat_swap()


if __name__ == "__main__":
    main()
