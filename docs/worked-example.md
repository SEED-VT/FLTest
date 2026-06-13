# Worked example: exhaustively testing attacks & defenses

This page walks through evaluating a federated-learning setup against a **matrix of attacks
and defenses from a single config**, then cross-checking it with differential, metamorphic,
and pitfall testing. All numbers below are real output from
`examples/configs/exhaustive_eval.yaml` (MNIST, MLP, 6 clients, 3 rounds, reference backend,
CPU).

## 1. One config, a whole scenario matrix

Each entry in `runs:` is an independent scenario with its **own** `attacks`/`defenses`
(per-run overrides); the shared settings are held fixed so scenarios are comparable.

```yaml
name: exhaustive_eval
dataset: mnist
model_name: MLP
num_clients: 6
num_rounds: 3
client_lr: 0.05
metrics: [accuracy, loss, per_client]

runs:
  - {framework: reference, name: baseline, attacks: []}
  - {framework: reference, name: label_flip,
     attacks: [{name: label_flip, params: {shift: 1}, target_clients: [0, 1]}]}
  - {framework: reference, name: sign_flip,
     attacks: [{name: sign_flip, params: {scale: 3.0}, target_clients: [0, 1]}]}
  - {framework: reference, name: gaussian,
     attacks: [{name: gaussian, params: {sigma: 0.5}, target_clients: [0, 1]}]}
  - {framework: reference, name: backdoor,
     attacks: [{name: backdoor, params: {target_label: 0, infection_rate: 0.8, patch_size: 5}, target_clients: [0, 1]}]}
  - {framework: reference, name: backdoor+median,    ...same backdoor..., defenses: [{name: median}]}
  - {framework: reference, name: backdoor+trimmed,   ...same backdoor..., defenses: [{name: trimmed_mean, params: {trim: 2}}]}
  - {framework: reference, name: backdoor+krum,      ...same backdoor..., defenses: [{name: krum, params: {num_byzantine: 2}}]}
  - {framework: reference, name: backdoor+normclip,  ...same backdoor..., defenses: [{name: norm_clip, params: {clip_norm: 0.5}}]}
```

Run it:

```bash
fltest run examples/configs/exhaustive_eval.yaml
```

## 2. Results

| Scenario | accuracy | attack_success_rate | per-client acc (mean / min) |
|----------|:--------:|:-------------------:|:---------------------------:|
| baseline | 0.8965 | – | 0.8934 / 0.8840 |
| label_flip | 0.8398 | – | 0.8178 / 0.8077 |
| sign_flip | **0.0156** | – | 0.0196 / 0.0143 |
| gaussian | 0.5430 | – | 0.5525 / 0.5377 |
| backdoor | 0.8887 | **0.4735** | 0.8800 / 0.8667 |
| backdoor + median | 0.8896 | **0.0205** | 0.8794 / 0.8697 |
| backdoor + trimmed_mean | 0.8896 | **0.0205** | 0.8794 / 0.8697 |
| backdoor + krum | 0.8301 | **0.0032** | 0.8214 / 0.8090 |
| backdoor + norm_clip | 0.8496 | 0.6562 | 0.8391 / 0.8247 |

## 3. What this tells you

- **Accuracy alone hides attacks.** The `backdoor` run keeps **0.889 accuracy** — it looks
  healthy — yet its **attack success rate is 0.47**. Without the ASR metric you would call
  this model "robust." This is exactly the over-estimation the project warns about.
- **"Naive" ≠ harmless.** `sign_flip` (2 of 6 clients) **collapses the model to 0.016**;
  `gaussian` halves accuracy. Undefended FedAvg is fragile.
- **Robust aggregation works.** `median` and `trimmed_mean` drop backdoor ASR from 0.47 to
  **0.02** with **no accuracy cost**; `krum` drives ASR to **0.003** at a small accuracy
  cost (0.83).
- **The right defense matters.** `norm_clip` *does not* help this backdoor (ASR 0.66) —
  clipping honest and malicious updates equally doesn't remove the trigger signal. FLTest
  surfaces this rather than letting you assume any defense helps.
- **Personalized view.** `per_client_acc_min` tracks the worst-off client; it falls under
  attack and recovers under robust defenses, exposing disparity a single global number hides.

Every value above is in `reports/exhaustive_eval_run.json` (per-round `history` + `final`).

## 4. Cross-framework differential check

Do the frameworks agree on a clean run? `examples/configs/differential_3way.yaml`:

```bash
fltest diff examples/configs/differential_3way.yaml
```

```
reference  acc=0.8838
flower     acc=0.8730
nvflare    acc=0.8672
[✓ PASS] differential: mnist/iid MLP c3 r2 :: ['reference', 'flwr', 'nvflare']
        max|Δ|=0.0166 (tol=0.1)
```

All three backends land within 0.017 — a divergence beyond tolerance would point to a
framework bug (the class of issue the project's AppFL CUDA-on-CPU finding represents).

## 5. Metamorphic check

`examples/configs/metamorphic.yaml`:

```bash
fltest metamorphic examples/configs/metamorphic.yaml
```

```
[✓ PASS] clients_scale (num_clients)    non-decreasing over [4.0, 8.0]
[✓ PASS] rounds_monotonic (num_rounds)  non-decreasing over [1.0, 3.0]
```

Doubling clients (IID) didn't drop accuracy, and more rounds didn't decrease it — the
relations that *should* hold, do.

## 6. Pitfall check on this very config

```bash
fltest pitfalls examples/configs/exhaustive_eval.yaml
```

```
[MEDIUM] MNIST-only evaluation       (P2_dataset)
[HIGH  ] IID-only data distribution  (P3_iid_only)
[MEDIUM] No privacy-leakage attack   (P5_subtle_leakage)
```

The checker reads the per-run attacks, so it does **not** complain about missing threat
models — but it correctly flags that this matrix is still MNIST-only, IID-only, and has no
privacy (DLG) attack, and prints copy-pasteable counter-experiments to fix each. Apply them
(e.g. `dataset: [mnist, cifar10]`, `data_distribution: [iid, dirichlet]`, add a `dlg`
attack) and the [fuzzer](fuzzing.md) expands the matrix accordingly.

## 7. Reproduce everything

```bash
conda activate fltest
pytest tests/ -q                                       # unit + smoke suite (green)
fltest run         examples/configs/exhaustive_eval.yaml
fltest diff        examples/configs/differential_3way.yaml
fltest metamorphic examples/configs/metamorphic.yaml
fltest pitfalls    examples/configs/exhaustive_eval.yaml
```

To add a scenario, append a run with your own attack/defense; to add a *new* attack or
defense, see **[Port your attacks & defenses](extending.md)**.
