# Metrics — what FLTest measures

Every run produces a `RunResult` with per-round `history` and a `final` dict (last round).
These metrics are what differential and metamorphic tests read.

## Always measured (every backend, every round)

Computed by evaluating the **global model on the central test set** after each round:

| Metric | Meaning |
|--------|---------|
| `accuracy` | top-1 accuracy of the global model on the test subset |
| `loss` | mean cross-entropy loss on the test subset |
| `gm_weight_sum` | sum of all global-model parameters — a cheap fingerprint to spot divergence/NaNs |

The test subset size is `max_test_data_size`. These three appear in `final` for every run.

## Produced by plugins (when configured)

| Metric | Produced by | Meaning |
|--------|-------------|---------|
| `attack_success_rate` | `backdoor` attack | fraction of a *triggered* test set predicted as the target label (excludes samples already of the target) |
| `reconstruction_mse` | `dlg` attack | pixel MSE between the reconstructed and true victim image (lower = better reconstruction = worse privacy) |
| `reconstruction_psnr` | `dlg` attack | peak signal-to-noise ratio of the reconstruction (higher = better reconstruction) |
| `label_recovery` | `dlg` attack | fraction of victim labels correctly recovered |
| `membership_inference_auc` | `membership_inference` attack | probability that a random training sample looks more member-like than a random held-out one; 0.5 is no leakage, 1.0 is perfect separation |
| `membership_loss_gap` | `membership_inference` attack | mean held-out loss minus mean training loss, which is the overfitting gap the attack exploits |
| `model_replacement_scale` | `model_replacement` attack | factor applied to the malicious client's delta from the current global model |
| `per_client_acc_mean` / `per_client_acc_min` | `per_client` listener | personalized accuracy of the final global model on each client's own data — `min` exposes representation disparity (project Pitfall-3) |

Add `per_client` to `metrics:` to enable personalized evaluation. Attack metrics appear
automatically when the relevant attack is configured.

### Secure aggregation

Masking either cancels exactly or it does not, and these metrics are how you tell which,
since a residue moves the model without necessarily moving accuracy.

| Metric | Produced by | Meaning |
|--------|-------------|---------|
| `secagg_mask_residual` | `secure_aggregation` | what the masks left behind after aggregation; 0 means they cancelled |
| `secagg_mask_to_update_ratio` | `secure_aggregation` | size of the mask relative to the update it hides; near 0 means the masking is cosmetic |
| `secagg_participants` | `secure_aggregation` | clients whose masked updates reached aggregation |
| `secagg_participant_mismatch` | `secure_aggregation` | clients that masked but never reached aggregation, so their masks never cancelled |
| `secagg_client_outside_mask_set` | `secure_aggregation` | a client masked against a peer set that did not include it |
| `mpc_agg_max_abs_error` | `mpc_aggregation` | largest absolute gap between the fixed-point aggregate and plain FedAvg |
| `mpc_agg_rel_error` | `mpc_aggregation` | that gap relative to the size of the aggregate |
| `mpc_overflow_rate` | `mpc_aggregation` | share of values that wrapped around the ring; above 0 means a silently wrong aggregate |
| `mpc_dropouts` | `mpc_aggregation` | clients removed after masking, whose pairwise masks stay in the sum |
| `mpc_max_encoded_bits` | `mpc_aggregation` | log2 of the largest encoded magnitude; above 53 the float64 encode drops low-order bits |

### Detection

| Metric | Produced by | Meaning |
|--------|-------------|---------|
| `fldetector_detected_count` | `fldetector` | how many clients were flagged and excluded |
| `fldetector_detected_clients` | `fldetector` | the client ids it flagged |
| `fldetector_scores` | `fldetector` | each client's suspicion score for that round |

`fldetector` records these in the **detection round**, not every round, so they appear in
`history` at that round rather than in `final`. The run matrix shows
`fldetector_detected_count` as a column when it is present. The other two are a list and a
mapping, so they stay in the JSON report rather than becoming fixed-width cells.

## Run parameters recorded beside the metrics

Each run's `params` in the JSON report carries its fully resolved settings, which includes
`aggregation`. That names `fedavg` or the robust rule that replaced it, since a
robust-aggregation defense substitutes the rule rather than perturbing an update.

## Where metrics live

- `result.history[round]` — dict of metrics for that round.
- `result.final` — metrics from the last round (what tests assert on).
- `result.extras` — non-scalar detail (e.g. the DLG reconstruction summary).
- JSON report under `reports/` contains all of the above.

## Which metric do the tests use?

Both testers operate on a **single scalar metric from `final`**, chosen by the config:

- **Differential** (`testing.differential.metric`, default `accuracy`): compares that
  metric across frameworks. See [Differential testing](differential-testing.md).
- **Metamorphic** (per-relation `metric`, default `accuracy`): tracks that metric as one
  input parameter is swept. See [Metamorphic testing](metamorphic-testing.md).

You can point either at any metric in `final` — e.g. set a metamorphic relation's
`metric: attack_success_rate` to assert that ASR is non-decreasing as attack strength rises,
or `metric: reconstruction_mse` to assert it rises as DP noise increases.
