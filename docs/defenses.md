# Defenses (PPFL techniques)

Defenses are hook plugins (`fltest/defenses/`) that subclass `PPFLBaseClass`. Declare them
in a config:

```yaml
defenses:
  - {name: <defense>, params: {...}}
```

Three flavors compose through the same hooks:

- **Client-side perturbation** acts at `after_client_train` on one client's update.
- **Robust aggregation** acts at `before_aggregate` by replacing the set of updates the
  backend will average.
- **Secure aggregation** blinds each update so the server sees only the sum. `secure_aggregation`
  masks client-side and lets FedAvg cancel the masks; `mpc_aggregation` simulates the whole
  fixed-point protocol at `before_aggregate`.

## Catalog

| Name | Type | Hook | Key params |
|------|------|------|-----------|
| `gradient_noise` | DP-style clip + Gaussian noise | `after_client_train` | `clip_norm` (1.0), `sigma` (0.01) |
| `norm_clip` | update-norm clipping | `after_client_train` | `clip_norm` (1.0) |
| `krum` | robust aggregation (select) | `before_aggregate` | `num_byzantine` (1) |
| `trimmed_mean` | robust aggregation (coordinate trim) | `before_aggregate` | `trim` (1) |
| `median` | robust aggregation (coordinate median) | `before_aggregate` | — |
| `secure_aggregation` | pairwise masking, float | `after_client_train` | `mask_scale` (1.0), `seed` (0) |
| `mpc_aggregation` | pairwise masking, fixed point | `before_aggregate` | `quant_bits` (16), `modulus` (2^32), `dropout_rate` (0.0) |
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

## Secure aggregation

Both variants build **pairwise masks** in the style of Bonawitz et al. (CCS 2017): every pair
of participants derives one shared pseudo-random vector, the lower-indexed party adds it and
the higher-indexed party subtracts it, so the masks telescope to zero once every participant's
contribution is summed. Each mask is a pure function of `(seed, round, client pair, layer)`, so
both parties derive it independently — which is what lets the Flower backend, where client
hooks run in separate Ray workers, use them at all.

!!! warning "What a single-process simulation can and cannot show"
    The masks come from a seed this process derives for both parties. There is no key
    agreement, no threshold secret sharing, and no dropout recovery, so **nothing here
    demonstrates cryptographic security**. What it does reproduce faithfully is the *simulated
    adversary view*: what an honest-but-curious server actually receives. Claims of the form
    "under masking, attack X no longer succeeds" are supported; claims of the form "secure
    aggregation protects Y" are not.

    Neither defense touches integer entries of the `state_dict` — BatchNorm's
    `num_batches_tracked`, a Hugging Face model's `position_ids`. A float mask cast back to
    int64 truncates, so the halves stop cancelling and the aggregate drifts silently. Those
    entries carry counters rather than learned information, so they pass through unmasked.

    Both defenses also assume the participant set is fixed for the round — masks cancel only
    across exactly the set that produced them. FLTest's backends use full participation, and
    `secure_aggregation` records `secagg_participant_mismatch` if a round ever aggregates a
    different number of updates rather than letting the residue pass silently.

**`secure_aggregation`** — float masking, no quantization. The client uploads
`x_i + m_i / n_i`; FedAvg's `sum_i (n_i / N) * x_i` then cancels the masks exactly. This is
the variant for privacy comparisons: the mask is in place at `before_aggregate`, which is
where the `dlg` attack with `source: shared_update` reads the uploaded update.

One consequence decides whether a configuration hides anything at all. The on-the-wire mask
has standard deviation `mask_scale * sqrt(P - 1) / n_i`, so **`mask_scale` is relative to the
client's shard size, not to the parameter scale** — a client with 5000 samples needs a
`mask_scale` three orders of magnitude above one with 5. Every round records
`secagg_mask_to_update_ratio`; a ratio near or below 1 means the masking is cosmetic.

Cancellation is exact in real arithmetic, but the wire format is float32, so a large
`mask_scale` leaves a rounding residue of roughly `1e-8 * mask_scale / 50` per coordinate.
That is the trade-off the parameter buys: stronger blinding, slightly noisier aggregate. For a
variant with no rounding to argue about, use `mpc_aggregation`.

**`mpc_aggregation`** — fixed-point masking in `Z_modulus`, which is how deployed secure
aggregation actually works, and which carries three failure modes a float simulation hides:

| Failure mode | Trigger | What to watch |
|---|---|---|
| Quantization error | too few `quant_bits` | `mpc_agg_max_abs_error` rises above the floor |
| Overflow | `modulus` too small for `sum_i n_i * x_i` | `mpc_overflow_rate` > 0; wraps silently |
| Masks not cancelling | `dropout_rate` > 0 | error jumps by orders of magnitude |

The whole protocol runs at `before_aggregate`, where the server holds every client's update.
That placement is what lets the defense compute plain FedAvg alongside the protocol's output
and record the exact gap as `mpc_agg_max_abs_error` — a strict numeric oracle rather than an
accuracy threshold. The cost is that the client-side view is not simulated: an attack reading
`ctx.updates_and_weights` still sees plaintext, because attacks attach before defenses on the
same hook. Use `secure_aggregation` for adversary-view experiments and this one for
arithmetic correctness.

`dropout_rate` drops clients *after* they have masked. FLTest does not implement the threshold
secret sharing that repairs this, so a run with dropouts is a run whose aggregate is knowingly
wrong — the parameter exists to measure the failure mode, not to survive it.

### Checking that the masks actually cancel

Masking is either lossless or broken, and an accuracy threshold is too coarse to tell the
difference. The `secagg_lossless` metamorphic relation is an **exact-equality** oracle: the
masks are supposed to cancel whatever they are, so changing the mask seed must leave the global
model bit-identical.

```yaml
testing:
  metamorphic:
    - {relation: secagg_lossless, parameter: defense.seed, values: [1, 2, 3],
       metric: gm_weight_sum, tolerance: 0.0}
```

Use `gm_weight_sum`, which fingerprints the model directly — `accuracy` rounds two different
models to the same number. The pitfall checker flags a secure-aggregation config that has no
such relation (`P4_untested_secagg`).

!!! note "Masking and robust aggregation are mutually exclusive"
    `secure_aggregation` hides individual updates; `krum` / `trimmed_mean` / `median` need to
    compare them client-by-client. A real server cannot do both. This simulation lets it,
    because the server holds plaintext either way — so the pitfall checker flags the
    combination (`P4_secagg_vs_robust`) rather than letting a config claim a defense stack
    nobody can deploy.

## Worked example: secure aggregation vs. gradient inversion

`examples/configs/secure_agg.yaml` runs `dlg` with `source: shared_update` — an
honest-but-curious server inverting the update it received — against three arms that differ
only in what the client uploads:

```bash
fltest run examples/configs/secure_agg.yaml
```

Compare `reconstruction_mse` across `none`, `gradient_noise`, and `secure_agg`; higher is a
worse reconstruction, i.e. a better defense. Check `secagg_mask_to_update_ratio` first — if it
is not comfortably above 1, the arm proves nothing and `mask_scale` needs raising.
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
