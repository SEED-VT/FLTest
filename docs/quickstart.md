# Quickstart

All commands assume `conda activate fltest` and that you are in the repo root.

## 1. See what's available

```bash
fltest list
```

## 2. Run an experiment matrix

```bash
fltest run examples/configs/differential.yaml
```

This runs the config on every framework in its `runs:` block (here: reference + Flower),
prints a table of final metrics, and writes a JSON report to `reports/`.

## 3. Differential test (frameworks must agree)

```bash
fltest diff examples/configs/differential_3way.yaml
```

Runs reference vs Flower vs NVFlare on the same config and checks their final accuracy is
within tolerance.

```
[✓ PASS] differential: mnist/iid MLP c3 r2 :: ['reference', 'flwr', 'nvflare']
        max|Δ|=0.0244 (tol=0.1)
```

## 4. Metamorphic test (relations must hold)

```bash
fltest metamorphic examples/configs/metamorphic.yaml
```

Checks e.g. that doubling clients (IID) does not drop accuracy and that more rounds does
not decrease it.

## 5. Attacks & defenses

```bash
fltest run examples/configs/attack_label_flip.yaml   # 2/5 clients flip labels
fltest run examples/configs/dlg.yaml                 # gradient-inversion privacy attack
fltest run examples/configs/defense_robust.yaml      # backdoor defeated by median aggregation
```

`defense_robust.yaml` shows attack-success-rate collapse from ~0.80 to ~0.03 while clean
accuracy is preserved.

## 6. Pitfall check + recommendations

```bash
fltest pitfalls examples/configs/pitfalls_demo.yaml
```

Flags MNIST-only / IID-only / no-attacks / no personalized metric and prints
copy-pasteable counter-experiments.

## 7. Loadable hook files (no config edits)

```bash
export FLTEST_HOOKS=examples/hooks/atk_dlg,examples/hooks/def_gradient_noise
fltest run examples/configs/dlg.yaml
```

Continue to the **[Approach walkthrough](walkthrough.md)** to see what happens end-to-end.
