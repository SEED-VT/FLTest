# Configuration reference

An experiment is a single `test_conf.yaml`. **Any top-level FL knob may be a scalar or a
list** — a list marks it for [fuzzing](fuzzing.md). The `runs:` block lists the frameworks
to execute (this is what enables cross-framework differential testing).

## Full example

```yaml
name: my_eval                    # report name
device: cpu                      # cpu | mps | cuda
seed: 786
deterministic: true

# ---- data ----
dataset: [mnist, cifar10]        # list => fuzzed
data_distribution: [iid, dirichlet]   # iid | dirichlet | pathological
dirichlet_alpha: 0.5             # lower = more non-IID (only for dirichlet)
classes_per_partition: 2         # only for pathological
dataset_partitions: 100          # how finely to shard before taking num_clients shards

# ---- model ----
model_name: LeNet                # LeNet | ConvNet | MLP

# ---- FL parameters ----
num_clients: 10
num_rounds: 10
client_epochs: 1
client_lr: 0.01
client_batch_size: 32
server_batch_size: 256
max_test_data_size: 2048
optimizer: SGD                   # SGD | Adam
loss_fn: CrossEntropyLoss

# ---- plugins ----
attacks:
  - {name: backdoor, params: {target_label: 0, infection_rate: 0.3}, target_clients: [0, 1]}
defenses:
  - {name: median}
metrics: [accuracy, loss, per_client]

# ---- which frameworks to run ----
runs:
  - {framework: reference, name: reference}
  - {framework: flwr,      name: flower}
  - {framework: nvflare,   name: nvflare}

# ---- what to assert ----
testing:
  differential: {enabled: true, mode: cross_framework, metric: accuracy, tolerance: 0.05}
  metamorphic:
    - {relation: clients_scale, values: [10, 20], metric: accuracy, tolerance: 0.05}
```

## Knob reference

### Reproducibility / hardware
| Key | Default | Notes |
|-----|---------|-------|
| `seed` | 786 | seeds python/numpy/torch |
| `device` | `cpu` | `cpu` is deterministic; `mps`/`cuda` for speed |
| `deterministic` | `true` | load cached identical initial weights for all clients/frameworks |
| `total_cpus` / `total_gpus` | 4 / 0 | Flower/Ray resource pool |

### Data
| Key | Default | Notes |
|-----|---------|-------|
| `dataset` | `mnist` | see [Datasets](datasets.md) |
| `data_distribution` | `iid` | `iid`, `dirichlet` (label skew), `pathological` (N classes/client) |
| `dirichlet_alpha` | 0.5 | only used when distribution is `dirichlet`; lower ⇒ more heterogeneous |
| `classes_per_partition` | 2 | only used when distribution is `pathological` |
| `dataset_partitions` | 100 | the dataset is split into this many shards; the first `num_clients` are used. Keep it fixed while sweeping `num_clients` so per-client data size is comparable. |
| `max_test_data_size` | 2048 | size of the central test subset (keeps eval fast) |

### Model
| Key | Default | Options |
|-----|---------|---------|
| `model_name` | `LeNet` | `LeNet` (32×32 conv), `ConvNet` (smooth activations, for DLG), `MLP` (fast) |

### FL parameters
| Key | Default | Notes |
|-----|---------|-------|
| `num_clients` | 10 | participants per round (all participate; `fraction_fit=1`) |
| `num_rounds` | 10 | global aggregation rounds |
| `client_epochs` | 1 | local epochs per round |
| `client_lr` | 0.01 | local learning rate |
| `client_batch_size` | 32 | local batch size |
| `optimizer` | `SGD` | `SGD`, `Adam` |

### Plugins
- `attacks:` / `defenses:` — lists of `{name, params, target_clients?}`. See
  [Attacks](attacks.md) and [Defenses](defenses.md) for each plugin's parameters.
  `target_clients` (attacks only) restricts the attack to those client ids; omit for all.
- `metrics:` — list of metric-listener names. `accuracy` and `loss` are always produced;
  add `per_client` for personalized evaluation. See [Metrics](metrics.md).

### `runs:`
A list of `{framework, name?, ...overrides}`. One entry per framework you want to execute.
A run entry may also override any top-level knob and may carry its own `attacks`/`defenses`/
`metrics`. Frameworks: `reference`, `flwr`/`flower`, `nvflare`/`flare`.

### `testing:`
- `differential:` — see [Differential testing](differential-testing.md).
- `metamorphic:` — a list of relations; see [Metamorphic testing](metamorphic-testing.md).

## Where parameters come from in code

Defaults and validation live in `fltest/core/config.py` (`TestConfig` and `RunSpec`). The
fuzzable knob list is `FUZZABLE_KNOBS` in the same file.
