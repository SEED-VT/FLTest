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

## How each works

**`gradient_noise`** — clips the client's update *delta* (relative to the current global
model) to `clip_norm`, then adds `N(0, sigma²)`. The user-space analogue of DP-SGD's
per-update clipping + noise. Sweep `sigma` to chart the privacy/utility trade-off (the
proposal's Pitfall-4).

**`norm_clip`** — clips the update delta's L2 norm to `clip_norm` without noise. Limits the
magnitude a malicious client can inject (mitigates scaled poisoning / sign-flip).

**`krum`** — selects the single client update closest to its `n − f − 2` nearest neighbours
(the most "agreed upon"), robust to up to `f = num_byzantine` adversaries.

**`trimmed_mean`** — for each coordinate, drops the `trim` largest and smallest values
across clients, then averages the rest.

**`median`** — coordinate-wise median across client updates. Simple and strong against a
Byzantine minority.

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
