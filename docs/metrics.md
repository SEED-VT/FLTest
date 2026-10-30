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
| `attack_success_rate` | `backdoor`, `little_is_enough_backdoor`, `little_is_enough_cropped_backdoor` attacks | fraction of a *triggered* test set predicted as the target label (excludes samples already of the target) |
| `reconstruction_mse` | `dlg` attack | pixel MSE between the reconstructed and true victim image (lower = better reconstruction = worse privacy) |
| `reconstruction_psnr` | `dlg` attack | peak signal-to-noise ratio of the reconstruction (higher = better reconstruction) |
| `label_recovery` | `dlg` attack | fraction of victim labels correctly recovered |
| `membership_inference_auc` | `membership_inference` attack | probability that a random training sample looks more member-like than a random held-out one; 0.5 is no leakage, 1.0 is perfect separation |
| `membership_loss_gap` | `membership_inference` attack | mean held-out loss minus mean training loss, which is the overfitting gap the attack exploits |
| `model_replacement_scale` | `model_replacement` attack | factor applied to the malicious client's delta from the current global model |
| `little_is_enough_z` | `little_is_enough` attack | standard deviations the crafted update sits from the benign mean; large enough to hurt and small enough to stay selected is the whole attack |
| `little_is_enough_drift` | `little_is_enough` attack | how far the round's aggregate actually landed from the benign mean, in benign standard deviations — the shift the attack really achieved |
| `little_is_enough_absorption` | `little_is_enough` attack | `drift / z`: the fraction of the requested shift the defense conceded. 0 means fully filtered, 1 means the crafted update was taken wholesale. Plain FedAvg concedes the attackers' share of total sample weight (`m/n` for equal shards), so anything above that line is a defense doing worse than no defense |
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

#### Was the detection correct?

The metrics above say what a detector flagged, not whether it was right. Accuracy does not
settle it either, because a detector that flagged one honest client alongside the attackers
would produce nearly the same recovery curve. Add the `detection` listener to `metrics:` and
the comparison is measured.

| Metric | Meaning |
|--------|---------|
| `detection_precision` | share of flagged clients that the config actually made malicious |
| `detection_recall` | share of the config's malicious clients that the defense flagged |
| `detection_f1` | harmonic mean of the two |
| `detection_false_positives` | honest clients the defense wrongly flagged |
| `detection_missed` | malicious clients the defense never flagged |
| `detection_accuracy` | DACC in the FLDetector paper: the share of all `num_clients` classified correctly, counting an unflagged client as classified benign |
| `detection_fpr` | share of honest clients flagged as malicious |
| `detection_fnr` | share of malicious clients left unflagged (`1 - detection_recall`) |

The last three are the FLDetector paper's own detection metrics, so a run can be read
against its Table 2 directly.

Ground truth is the union of `target_clients` over the attacks that make a client
malicious, which are `backdoor`, `label_flip`, `sign_flip`, `gaussian`,
`model_replacement`, `little_is_enough`, `little_is_enough_backdoor`, and
`little_is_enough_cropped_backdoor`. `dlg` and `membership_inference` are deliberately excluded, because an
honest-but-curious server names a *victim* rather than an adversary, and counting one would
mark an honest client as a detection the defense owed you.

The listener records nothing when a run has no detector, or when an attack leaves
`target_clients` unset and every client is therefore malicious, so adding it to any config
is safe. `examples/configs/fldetector.yaml` enables it, and the `fldetector` arm reports
precision 1.0000 and recall 1.0000 against zero false positives, while the `undefended` and
`median` arms show a dash.

A zero-division follows scikit-learn's `zero_division=0`: a detector that flagged nobody
scores 0 rather than a vacuous 1.

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
