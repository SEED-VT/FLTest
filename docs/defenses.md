# Defenses (PPFL techniques)

Defenses are hook plugins (`fltest/defenses/`) that subclass `PPFLBaseClass`. Declare them
in a config:

```yaml
defenses:
  - {name: <defense>, params: {...}}
```

Two flavors compose through the same hooks:

- **Client-side perturbation** acts at `after_client_train` on one client's update.
- **Robust aggregation** acts at `before_aggregate` by replacing the set of updates the
  backend will average.

## Catalog

| Name | Type | Hook | Key params |
|------|------|------|-----------|
| `gradient_noise` | DP-style clip + Gaussian noise | `after_client_train` | `clip_norm` (1.0), `sigma` (0.01) |
| `norm_clip` | update-norm clipping | `after_client_train` | `clip_norm` (1.0) |
| `krum` | robust aggregation (select) | `before_aggregate` | `num_byzantine` (1) |
| `trimmed_mean` | robust aggregation (coordinate trim) | `before_aggregate` | `trim` (1) |
| `median` | robust aggregation (coordinate median) | `before_aggregate` | — |
| `fldetector` | history-based client filtering | `before_round`, `before_aggregate`, `after_aggregate` | `window_size` (10), `start_round` (50), `max_clusters` (10), `gap_samples` (20) |

## How each works

**`gradient_noise`** — clips the client's update *delta* (relative to the current global
model) to `clip_norm`, then adds `N(0, sigma²)`. The user-space analogue of DP-SGD's
per-update clipping + noise. Sweep `sigma` to chart the privacy/utility trade-off (the
project's Pitfall-4).

**`norm_clip`** — clips the update delta's L2 norm to `clip_norm` without noise. Limits the
magnitude a malicious client can inject (mitigates scaled poisoning / sign-flip).

**`krum`** — selects the single client update closest to its `n − f − 2` nearest neighbours
(the most "agreed upon"), robust to up to `f = num_byzantine` adversaries.

**`trimmed_mean`** — for each coordinate, drops the `trim` largest and smallest values
across clients, then averages the rest.

**`median`** — coordinate-wise median across client updates. Simple and strong against a
Byzantine minority.

**`fldetector`** — compares each client's model delta with a limited-memory BFGS
prediction from earlier rounds. It scores inconsistencies over `window_size` rounds and
uses gap statistics and two-cluster k-means to identify the high-score group. Identified
clients are removed from the current aggregation and excluded from later rounds. This is
an *online* variant: unlike the [FLDetector paper](https://doi.org/10.1145/3534678.3539231),
it does not restart training after detection. It requires unique stable client IDs and
full client participation until detection; custom Flower clients must report their `cid`.
The default `start_round=50` follows the paper's warm-up choice, so shorter experiments
should lower it. At least `window_size + 2` rounds are needed to form the history.

To combine detection with a robust rule, place it first:

```yaml
defenses:
  - {name: fldetector, params: {window_size: 10, start_round: 50}}
  - {name: median}
```

`fldetector_scores`, `fldetector_detected_clients`, and
`fldetector_detected_count` are recorded in the detection round's metrics. The detector
fails explicitly if submission IDs are missing or an earlier hook changes the update
list's alignment. It is not available on NVFlare.

!!! note "Backend support"
    Client-side and robust-aggregation defenses run on the **reference** and **Flower**
    backends. **NVFlare** runs clients in separate processes, so it does not apply
    client-side hooks (it's used for cross-framework parity of vanilla FedAvg).

## Worked example: defeating a backdoor

`examples/configs/defense_robust.yaml` — two of six clients run a strong backdoor;
`median` aggregation rejects the poisoned updates:

| Defense | attack_success_rate | accuracy |
|---------|:------------------:|:--------:|
| none | 0.80 | 0.90 |
| `median` | 0.03 | 0.90 |
| `norm_clip` (clip_norm 0.5) | 0.67 | 0.86 |

```yaml
attacks:  [{name: backdoor, params: {infection_rate: 0.8, patch_size: 5}, target_clients: [0, 1]}]
defenses: [{name: median}]
metrics:  [accuracy, loss, per_client]
```

## Sweep a defense parameter (metamorphic)

```yaml
defenses: [{name: gradient_noise, params: {clip_norm: 1.0, sigma: 0.05}}]
testing:
  metamorphic:
    - {relation: dp_noise, parameter: defense.sigma, values: [0.0, 0.05, 0.1, 0.2], metric: accuracy}
```

More noise should not *increase* accuracy (utility non-increasing). See
**[Metamorphic testing](metamorphic-testing.md)**.

To add your own defense, see **[Port your attacks & defenses](extending.md)**.
