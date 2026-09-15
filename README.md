<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)"
            srcset="docs/assets/png/fltest-lockup-on-maroon-1600.png">
    <img src="docs/assets/png/fltest-lockup-1600.png" alt="FLTest" width="430">
  </picture>
</p>

**A testbed for evaluating the privacy and robustness of Privacy-Preserving Federated
Learning (PPFL).** FLTest gives *software-defined control and visibility* into FL testing.
One YAML config runs the same experiment across several FL frameworks, injects attacks and
defenses as composable hooks, and applies differential testing, metamorphic testing, and a
pitfall checker.

FLTest is the **NSF PDaSP (Track 3) FLTEST project** — *A Testbed for Enhancing Privacy and
Robustness of Federated Learning Systems* — supported by the U.S. National Science Foundation
(**Award #2452817-19**).

<p align="center">
  <img src="docs/assets/logos/nsf.png"   alt="U.S. National Science Foundation"        height="64" hspace="20" vspace="6">
  <img src="docs/assets/logos/umn.png"   alt="University of Minnesota"                 height="42" hspace="20" vspace="6">
  <img src="docs/assets/logos/vt.png"    alt="Virginia Tech"                           height="50" hspace="20" vspace="6">
  <img src="docs/assets/logos/umass.png" alt="University of Massachusetts Amherst"     height="34" hspace="20" vspace="6">
</p>

**Principal Investigators:**
[Ali Anwar](https://chalianwar.github.io/) (University of Minnesota) ·
[Muhammad Ali Gulzar](https://people.cs.vt.edu/~gulzar/) (Virginia Tech) ·
[Fatima Anwar](https://people.umass.edu/fanwar/) (University of Massachusetts Amherst)

## Why

A survey of 50 FL robustness papers found wildly inconsistent setups (MNIST-only, IID-only,
naive attacks, no personalized metrics), which inflates privacy/robustness claims. FLTest
makes a rigorous setup the default and *checks* for the common pitfalls.

## Highlights

- **One abstraction, many backends.** Every FL framework implements a single
  `run_simulation()` adapter. Built in: a dependency-light **reference** PyTorch FedAvg
  oracle, **Flower**, and **NVFlare** (optional extra).
- **Everything is a hook.** Attacks, defenses, and metric listeners are hook plugins that
  share one `HookContext`. A plugin written once runs across every backend, and several
  plugins compose on a single run.
- **Attacks:** `label_flip`, `sign_flip`, `gaussian`, `backdoor` (with attack-success-rate),
  `dlg` (gradient-inversion privacy attack), `membership_inference` (loss-threshold
  privacy attack, scored as AUC each round).
- **Defenses (PPFL):** `gradient_noise` (DP-style clip+noise), `norm_clip`, robust
  aggregation `krum` / `trimmed_mean` / `median`, and secure aggregation
  `secure_aggregation` (float pairwise masking) / `mpc_aggregation` (fixed point in a
  finite ring, which surfaces quantization, overflow, and dropout errors).
- **Differential testing:** same config across frameworks must agree within tolerance
  (cross-framework parity); or the same spec run twice must be identical (determinism).
- **Metamorphic testing:** `clients_scale` (N→2N), `rounds_monotonic`, `attack_strength`,
  `dp_noise`, and `secagg_lossless` — an exact-equality oracle: a secure-aggregation mask
  seed must leave the global model bit-identical.
- **Pitfall checker + recommender:** flags the six FL-evaluation pitfalls from the project
  and emits copy-pasteable counter-experiments.
- **Config fuzzer:** any list-valued knob (e.g. `dataset: [mnist, cifar10]`) is expanded
  into a grid of runs.
- **Datasets:** `mnist`, `fashion_mnist`, `cifar10`, `cifar100`, and `femnist`, which is
  partitioned by writer and naturally non-IID. Text is supported through `ag_news`, and any
  Hugging Face image-classification id also works by name.
- **Models:** built-in `LeNet` / `ConvNet` / `MLP`, torchvision architectures such as
  `ResNet18` and `MobileNetV3`, and Hub models written as `hf:<id>`.

## Install (isolated conda env)

```bash
conda env create -f environment.yml      # creates env "fltest" (Python 3.11)
conda activate fltest
pip install -e ".[dev]"                   # core (reference + Flower) + test tooling
pip install -e ".[nvflare]"               # optional NVFlare backend (needs Python <=3.11)
pip install -e ".[hf]"                    # optional Hugging Face models and text datasets
```

CPU is the default and is deterministic; `device: mps` (Apple Silicon) or `device: cuda`
are selectable for speed (with the usual GPU non-determinism caveat).

An Intel Mac constrains the Hugging Face extra, because PyTorch publishes no macOS x86_64
wheel past 2.2.2. See [installation](docs/installation.md) for what that pins.

## Use

```bash
fltest list          # available frameworks/attacks/defenses/metrics

# the attack and defense matrix, on CIFAR-10, with two privacy scenarios
fltest run examples/configs/exhaustive_eval.yaml

# privacy: what the model leaks, and what DP noise costs to stop it
fltest run examples/configs/membership_inference.yaml
fltest run examples/configs/dlg.yaml               # gradient inversion
fltest run examples/configs/secure_agg.yaml        # DLG vs no defense / DP noise / masking

# heterogeneity: one client per real writer, and one topic per client
fltest run examples/configs/femnist_natural.yaml
fltest run examples/configs/text_domains.yaml      # needs the [hf] extra

# the testing engines
fltest diff        examples/configs/differential_cifar10_3way.yaml
fltest metamorphic examples/configs/metamorphic.yaml
fltest pitfalls    examples/configs/pitfalls_demo.yaml
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

- **NVFlare** accepts built-in models only, since it rebuilds the model from its class
  path and cannot serialise a torchvision constructor argument.
- **NVFlare** runs each client in its own simulator process, so client-side hooks at
  `before_client_train` and `after_client_train` do not apply to it. That backend is used
  for cross-framework differential parity of the vanilla FedAvg path. The reference and Flower
  backends support the full hook surface.
- **DLG** `source: gradient` (default) demonstrates raw-gradient invertibility. The
  `source: shared_update` mode (reconstruct from the uploaded update) is faithful only
  under single-step (FedSGD) training.
- A `Dockerfile` (CPU, Linux) is provided as a deliverable; the verified path is the conda
  env above.

## Acknowledgement

This material is based upon work supported by the **U.S. National Science Foundation** under
the **Privacy-preserving Data Sharing in Practice (PDaSP) program, Track 3 — Usable Tools and
Testbeds for Confidential Data Sharing**, **Award #2452817-19**. The PDaSP program is supported
by the NSF together with its co-sponsors (U.S. Department of Transportation, Intel, NIST, and
Broadcom). Any opinions, findings, and conclusions or recommendations expressed in this
material are those of the authors and do not necessarily reflect the views of the National
Science Foundation or its co-sponsors. Program information: <https://pdasp.net/projects/>.
