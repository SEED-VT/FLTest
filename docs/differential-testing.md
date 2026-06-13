# Differential testing

**Idea:** runs that differ *only* in some controlled way should agree. FLTest supports two
modes (`testing.differential.mode`). Code: `fltest/testing/differential.py`.

## What is measured

A single scalar metric from each run's `final` dict, selected by
`testing.differential.metric` (default `accuracy`). Typically the final-round global-model
test accuracy.

## Mode 1: `cross_framework` (parity)

Same logical config, different FL framework ⇒ same final metric within tolerance. This
surfaces framework-implementation divergences (the kind of bug the proposal's AppFL
CUDA-on-CPU finding represents).

**How it groups:** runs are grouped by a *logical key* — every spec field except the
framework (`dataset`, `data_distribution`, `model_name`, `num_clients`, `num_rounds`,
`client_epochs`, `client_lr`, `client_batch_size`, `optimizer`, `seed`, and the
attack/defense set). Within each group of ≥2 frameworks:

**Rule (`parity`):** `max(metric) − min(metric) ≤ tolerance` ⇒ PASS.

```yaml
runs:
  - {framework: reference}
  - {framework: flwr}
  - {framework: nvflare}
testing:
  differential: {mode: cross_framework, metric: accuracy, tolerance: 0.05}
```

```
[✓ PASS] differential: mnist/iid MLP c3 r2 :: ['reference', 'flwr', 'nvflare']
        max|Δ|=0.0244 (tol=0.1)
```

!!! note "Parity, not bit-identity"
    Different frameworks use different RNG streams and execution paths, so exact equality is
    impossible. The reference and Flower backends share FLTest's own weighted-mean
    aggregation, which keeps them close; the tolerance (default 0.05, as in the proposal
    slides) absorbs the residual. Choose tolerance to match your metric's natural variance.

## Mode 2: `determinism`

The same spec run twice must give an (almost) identical result — surfaces hidden
nondeterminism / state leakage.

**Rule:** `|metric(run1) − metric(run2)| ≤ tolerance` (use a tiny `tolerance`, e.g. `1e-4`),
on `device: cpu`.

```yaml
testing:
  differential: {mode: determinism, metric: accuracy, tolerance: 0.0001}
```

## Run it

```bash
fltest diff <config.yaml>
```

Prints a PASS/FAIL table, writes `reports/<name>_differential.json`, and exits non-zero if
any check fails (CI-friendly).
