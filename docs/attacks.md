# Attacks

Attacks are hook plugins (`fltest/attacks/`) that subclass `ThreatModelBaseClass`. Declare
them in a config:

```yaml
attacks:
  - {name: <attack>, params: {...}, target_clients: [0, 1]}   # target_clients optional (default: all)
```

Multiple attacks compose. `target_clients` restricts which clients are adversarial.

`backdoor` and `dlg` operate on pixels, so they apply only to image datasets and raise a
clear error on a text run. `label_flip`, `sign_flip`, `gaussian`, and `model_replacement`
work on labels or updates, so they apply to either modality.

## Catalog

| Name | Type | Hook(s) | Key params |
|------|------|---------|-----------|
| `label_flip` | data poisoning | `before_client_train` | `shift` (default 1), `mapping` |
| `gaussian` | model poisoning (naive) | `after_client_train` | `sigma` (0.1) |
| `sign_flip` | model poisoning | `after_client_train` | `scale` (1.0) |
| `model_replacement` | model poisoning (targeted) | `after_client_train` | `scale` (automatic), `target_round` |
| `backdoor` | data poisoning (targeted) | `before_client_train`, `after_round` | `target_label` (0), `infection_rate` (0.3), `patch_size` (4), `patch_value` (1.0) |
| `dlg` | privacy (gradient inversion) | `before_client_train` (+ `before_aggregate` in shared_update mode) | `target_client`, `target_round`, `num_images`, `iters`, `source` |
| `membership_inference` | privacy (inference) | `on_data_distribute`, `after_round` | `target_client` (0), `max_samples` (512) |

## How each works

**`label_flip`** — wraps the attacker's loader and relabels each batch
(`y → (y+shift) % num_classes`, or a fixed `{src: dst}` mapping). A classic robustness
attack; weak alone (one of the "naive" attacks the project flags).

**`gaussian`** — adds zero-mean Gaussian noise to the attacker's update
(`u' = u + N(0, sigma²)`). Naive Byzantine attack; useful as a baseline.

**`sign_flip`** — reflects the update around the global model and scales it
(`u' = g − scale·(u − g)`), pushing aggregation in the opposite direction.

**`model_replacement`** — boosts a malicious local model around the current global model
(`u' = g + scale·(u − g)`) so the malicious delta survives aggregation. This implements
the train-and-scale attack from Bagdasaryan et al., [*How To Backdoor Federated
Learning*](https://proceedings.mlr.press/v108/bagdasaryan20a.html). Compose it after
`backdoor` to boost a locally learned trigger, and set `target_round` for a single-shot
attack. Without an explicit `scale`, FLTest uses `num_clients / num_attackers`, which is
the replacement factor for full-participation, equal-weight FedAvg.

The hook interface exposes the local model, global model, client identity, and round, so
the attack itself is backend-neutral on reference and Flower. It does not expose the
round's total sample weight to a client, however. Automatic scaling is therefore only an
estimate when clients have unequal sample weights; pass the exact factor explicitly in
that case. NVFlare does not run client-side hooks and cannot apply this attack.

The runnable comparison uses one of four clients as the attacker and boosts its backdoored
model only in round 3, at an automatic scale of 4.0. Replacement takes attack success from
0.2930 to 1.0000 on the reference backend and from 0.2724 to 0.9892 on Flower, costing
0.0196 and 0.0361 of clean accuracy:

| Backend | Attack | ASR | Accuracy |
|---------|--------|----:|---------:|
| reference | backdoor only | 0.2930 | 0.8926 |
| reference | model replacement | 1.0000 | 0.8730 |
| Flower | backdoor only | 0.2724 | 0.8867 |
| Flower | model replacement | 0.9892 | 0.8506 |

**`backdoor`** — the attacker stamps a bright patch on a fraction (`infection_rate`) of its
images and relabels them to `target_label`; the global model learns
*trigger ⇒ target_label*. At each round end it measures **attack success rate (ASR)** — the
fraction of a triggered test set predicted as the target — and records it as a metric. This
is the headline robustness signal; pair it with a robust-aggregation defense to see ASR drop
(see [Defenses](defenses.md)).

**`membership_inference`** — asks whether a given record was in a client's training data,
which is the canonical privacy attack the proposal cites and the one Pitfall-1 says
evaluations skip. It models an honest-but-curious server that sees the global model each
round. The score is the per-sample loss, following Yeom et al., since a model assigns lower
loss to data it trained on. Members are the target client's training data and non-members
are the held-out test set. It records `membership_inference_auc`, where 0.5 means no
leakage and 1.0 means members and non-members separate perfectly, alongside
`membership_loss_gap`. No shadow model is needed, and because it reads only losses it
applies to text as readily as to images.

For example, `examples/configs/membership_inference.yaml` runs the same overfitted setup
twice. Undefended it reaches AUC 0.67, and clipping with Gaussian noise takes it to 0.50,
which is chance, for about six points of accuracy.

**`dlg`** — Deep Leakage from Gradients: reconstructs a victim client's private batch by
optimizing a dummy batch so its gradient matches the victim's. Records `reconstruction_mse`,
`reconstruction_psnr`, and `label_recovery`. Use `model_name: ConvNet` (smooth activations)
on `device: cpu`. Two threat sources:

- `source: gradient` (default) — reconstruct from the *raw per-step gradient*; demonstrates
  pure invertibility.
- `source: shared_update` — reconstruct from the *uploaded (post-defense) update*; faithful
  only under single-step (FedSGD) training.

## Examples

```yaml
# label flip on two clients
attacks: [{name: label_flip, params: {shift: 1}, target_clients: [0, 1]}]
```

```yaml
# backdoor measured by ASR
attacks: [{name: backdoor, params: {target_label: 0, infection_rate: 0.8, patch_size: 5}, target_clients: [0, 1]}]
```

```yaml
# boost one backdoored client in round 3; scale defaults to num_clients
attacks:
  - {name: backdoor, params: {target_label: 0, infection_rate: 0.8}, target_clients: [0]}
  - {name: model_replacement, params: {target_round: 3}, target_clients: [0]}
```

```yaml
# privacy attack
model_name: ConvNet
dataset: cifar10
attacks: [{name: dlg, params: {target_client: 0, target_round: 1, iters: 300, source: gradient}}]
```

Runnable: `examples/configs/attack_label_flip.yaml`, `model_replacement.yaml`, `dlg.yaml`.

To add your own attack, see **[Port your attacks & defenses](extending.md)**.
