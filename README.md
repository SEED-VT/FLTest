# FLTest

**A testbed for evaluating the privacy and robustness of Privacy-Preserving Federated
Learning (PPFL).** FLTest gives *software-defined control and visibility* into FL testing:
run the same experiment across multiple FL frameworks, inject attacks and defenses as
composable hooks, and apply **differential** and **metamorphic** tests plus a **pitfall
checker** — all from a single YAML config.

This is the implementation of the NSF PDaSP Track-3 FLTEST proposal.

## Why

A survey of 50 FL robustness papers found wildly inconsistent setups (MNIST-only, IID-only,
naive attacks, no personalized metrics), which inflates privacy/robustness claims. FLTest
makes a rigorous setup the default and *checks* for the common pitfalls.

## Highlights

- **One abstraction, many backends.** Every FL framework implements a single
  `run_simulation()` adapter. Built in: a dependency-light **reference** PyTorch FedAvg
  oracle, **Flower**, and **NVFlare** (optional extra).
- **Everything is a hook.** Attacks, defenses, and metric listeners are hook plugins that
  share one `HookContext`, so a plugin written once runs across every backend and multiple
  plugins compose on a single run.
- **Attacks:** `label_flip`, `sign_flip`, `gaussian`, `backdoor` (with attack-success-rate),
  `dlg` (gradient-inversion privacy attack).
- **Defenses (PPFL):** `gradient_noise` (DP-style clip+noise), `norm_clip`, and robust
  aggregation `krum` / `trimmed_mean` / `median`.
- **Differential testing:** same config across frameworks must agree within tolerance
  (cross-framework parity); or the same spec run twice must be identical (determinism).
- **Metamorphic testing:** `clients_scale` (N→2N), `rounds_monotonic`, `attack_strength`,
  `dp_noise` relations.
- **Pitfall checker + recommender:** flags the six FL-evaluation pitfalls from the proposal
  and emits copy-pasteable counter-experiments.
- **Config fuzzer:** any list-valued knob (e.g. `dataset: [mnist, cifar10]`) is expanded
  into a grid of runs.

## Install (isolated conda env)

```bash
conda env create -f environment.yml      # creates env "fltest" (Python 3.11)
conda activate fltest
pip install -e ".[dev]"                   # core (reference + Flower) + test tooling
pip install -e ".[nvflare]"               # optional NVFlare backend (needs Python <=3.11)
```

CPU is the default and is deterministic; `device: mps` (Apple Silicon) or `device: cuda`
are selectable for speed (with the usual GPU non-determinism caveat).

## Use

```bash
fltest list                                              # available frameworks/attacks/defenses/metrics
fltest run         examples/configs/differential.yaml
fltest diff        examples/configs/differential_3way.yaml   # cross-framework parity
fltest metamorphic examples/configs/metamorphic.yaml
fltest pitfalls    examples/configs/pitfalls_demo.yaml
fltest run         examples/configs/attack_label_flip.yaml
fltest run         examples/configs/dlg.yaml                  # privacy attack
fltest run         examples/configs/defense_robust.yaml      # backdoor vs median agg
```

Loadable hook files (slide-style), no config edits:

```bash
export FLTEST_HOOKS=examples/hooks/atk_dlg,examples/hooks/def_gradient_noise
fltest run examples/configs/dlg.yaml
```

## Config sketch (`test_conf.yaml`)

```yaml
name: my_eval
dataset: [mnist, cifar10]        # a list => fuzzed into a grid
data_distribution: [iid, dirichlet]
model_name: LeNet
num_clients: 10
num_rounds: 10
attacks:  [{name: backdoor, params: {infection_rate: 0.3}, target_clients: [0,1]}]
defenses: [{name: median}]
metrics:  [accuracy, loss, per_client]
runs:                            # one per framework => cross-framework differential
  - {framework: reference}
  - {framework: flwr}
  - {framework: nvflare}
testing:
  differential: {mode: cross_framework, metric: accuracy, tolerance: 0.05}
  metamorphic:
    - {relation: clients_scale, values: [10, 20], tolerance: 0.05}
```

## Tests

```bash
pytest tests/ -q
```

See `docs/ARCHITECTURE.md` for the design and `examples/configs/` for runnable configs.

## Notes & limitations

- **NVFlare** runs each client in its own simulator process, so client-side hooks
  (attacks/defenses at `before/after_client_train`) do not apply to it — it is used for
  cross-framework differential parity of the vanilla FedAvg path. The reference and Flower
  backends support the full hook surface.
- **DLG** `source: gradient` (default) demonstrates raw-gradient invertibility. The
  `source: shared_update` mode (reconstruct from the uploaded update) is faithful only
  under single-step (FedSGD) training.
- A `Dockerfile` (CPU, Linux) is provided as a deliverable; the verified path is the conda
  env above.
