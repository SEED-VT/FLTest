# Attacks

Attacks are hook plugins (`fltest/attacks/`) that subclass `ThreatModelBaseClass`. Declare
them in a config:

```yaml
attacks:
  - {name: <attack>, params: {...}, target_clients: [0, 1]}   # target_clients optional (default: all)
```

Multiple attacks compose. `target_clients` restricts which clients are adversarial.

## Catalog

| Name | Type | Hook(s) | Key params |
|------|------|---------|-----------|
| `label_flip` | data poisoning | `before_client_train` | `shift` (default 1), `mapping` |
| `gaussian` | model poisoning (naive) | `after_client_train` | `sigma` (0.1) |
| `sign_flip` | model poisoning | `after_client_train` | `scale` (1.0) |
| `backdoor` | data poisoning (targeted) | `before_client_train`, `after_round` | `target_label` (0), `infection_rate` (0.3), `patch_size` (4), `patch_value` (1.0) |
| `dlg` | privacy (gradient inversion) | `before_client_train` (+ `before_aggregate` in shared_update mode) | `target_client`, `target_round`, `num_images`, `iters`, `source` |

## How each works

**`label_flip`** — wraps the attacker's loader and relabels each batch
(`y → (y+shift) % num_classes`, or a fixed `{src: dst}` mapping). A classic robustness
attack; weak alone (one of the "naive" attacks the project flags).

**`gaussian`** — adds zero-mean Gaussian noise to the attacker's update
(`u' = u + N(0, sigma²)`). Naive Byzantine attack; useful as a baseline.

**`sign_flip`** — reflects the update around the global model and scales it
(`u' = g − scale·(u − g)`), pushing aggregation in the opposite direction.

**`backdoor`** — the attacker stamps a bright patch on a fraction (`infection_rate`) of its
images and relabels them to `target_label`; the global model learns
*trigger ⇒ target_label*. At each round end it measures **attack success rate (ASR)** — the
fraction of a triggered test set predicted as the target — and records it as a metric. This
is the headline robustness signal; pair it with a robust-aggregation defense to see ASR drop
(see [Defenses](defenses.md)).

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
# privacy attack
model_name: ConvNet
dataset: cifar10
attacks: [{name: dlg, params: {target_client: 0, target_round: 1, iters: 300, source: gradient}}]
```

Runnable: `examples/configs/attack_label_flip.yaml`, `dlg.yaml`.

To add your own attack, see **[Port your attacks & defenses](extending.md)**.
