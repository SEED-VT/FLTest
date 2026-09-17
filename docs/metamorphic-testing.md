# Metamorphic testing

**Idea:** transform one input parameter and assert an *expected relation* on the output
metric — no ground-truth label needed. Code: `fltest/testing/metamorphic.py`.

## What is measured

For each relation, FLTest sweeps one parameter over `values` and runs the simulation at
each value, using the first framework in `runs:`. It then reads one scalar `metric` from
each run's `final` (default `accuracy`) and applies a monotonicity rule with a
`tolerance`.

To isolate the swept parameter, any *other* list-valued knob is collapsed to its first value
("scalarized") for these runs.

## Built-in relations

| `relation` | Parameter swept | Rule on metric | Intuition |
|-----------|-----------------|----------------|-----------|
| `clients_scale` | `num_clients` | non-decreasing | doubling clients (IID, comparable per-client size) shouldn't drop accuracy |
| `rounds_monotonic` | `num_rounds` | non-decreasing | more rounds shouldn't decrease accuracy |
| `attack_strength` | *(you specify)* | non-increasing | stronger attack shouldn't *raise* accuracy |
| `dp_noise` | *(you specify)* | non-increasing | more privacy noise shouldn't raise accuracy (utility) |
| `secagg_lossless` | `defense.seed` | exactly equal | a secure-aggregation mask seed must not move the result *at all* |

`non_decreasing`: each step may rise or stay; a drop greater than `tolerance` fails.
`non_increasing`: the mirror image. (Rules: `fltest/testing/rules.py`.)

`exactly_equal` is the odd one out. The other three are inequalities with slack, which is
all an accuracy-based oracle can support: they catch a defense that is *badly* wrong.
`secagg_lossless` applies where the transform provably must not move the result at all.
Secure aggregation's masks are supposed to cancel whatever they are, so a different mask
seed must leave the global model bit-identical. Give it `tolerance: 0.0`, and a metric that
fingerprints the model rather than one that projects it through a test set. `gm_weight_sum`
does that, where `accuracy` rounds two different models to the same number:

```yaml
defenses: [{name: secure_aggregation, params: {mask_scale: 5000.0, seed: 1}}]
testing:
  metamorphic:
    - {relation: secagg_lossless, parameter: defense.seed, values: [1, 2, 3],
       metric: gm_weight_sum, tolerance: 0.0}
```

Note that `secure_aggregation` masks in float32, so a large `mask_scale` leaves a rounding
residue; if this fails by a hair, lower `mask_scale` before suspecting the protocol.
`mpc_aggregation` cancels in a finite ring and has no rounding to argue about. See
[Defenses](defenses.md#secure-aggregation).

## Sweeping plugin parameters

For `attack_strength` and `dp_noise`, point `parameter` at an attack or defense parameter
using a dotted path. `attack.<param>` targets the first attack and `defense.<param>` the
first defense:

```yaml
attacks: [{name: backdoor, params: {infection_rate: 0.1}}]
testing:
  metamorphic:
    # stronger backdoor must not increase clean accuracy
    - {relation: attack_strength, parameter: attack.infection_rate,
       values: [0.1, 0.3, 0.6], metric: accuracy, tolerance: 0.05}
```

```yaml
defenses: [{name: gradient_noise, params: {sigma: 0.0}}]
testing:
  metamorphic:
    # more DP noise must not increase accuracy
    - {relation: dp_noise, parameter: defense.sigma,
       values: [0.0, 0.05, 0.1, 0.2], metric: accuracy, tolerance: 0.05}
```

You can also assert on non-accuracy metrics, e.g. `metric: attack_success_rate` with a
non-decreasing expectation as attack strength grows (write that as a custom relation or use
`attack_strength` with the appropriate metric and read the detail).

## Example

```yaml
runs: [{framework: reference}]
testing:
  metamorphic:
    - {relation: clients_scale, values: [4, 8], metric: accuracy, tolerance: 0.1}
    - {relation: rounds_monotonic, values: [1, 3], metric: accuracy, tolerance: 0.1}
```

## Run it

```bash
fltest metamorphic <config.yaml>
```

```
[✓ PASS] metamorphic: clients_scale (num_clients, metric=accuracy)
        non-decreasing over [4.0, 8.0]
[✓ PASS] metamorphic: rounds_monotonic (num_rounds, metric=accuracy)
        non-decreasing over [1.0, 3.0]
```

Writes `reports/<name>_metamorphic.json` and exits non-zero on any FAIL.
