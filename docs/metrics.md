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
| `per_client_acc_mean` / `per_client_acc_min` | `per_client` listener | personalized accuracy of the final global model on each client's own data — `min` exposes representation disparity (proposal Pitfall-3) |

Add `per_client` to `metrics:` to enable personalized evaluation. Attack metrics appear
automatically when the relevant attack is configured.

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
