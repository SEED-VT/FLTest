# Config fuzzing

FLTest's orchestration includes a **config fuzzer**. Any top-level knob written as a
*list* is expanded, and the cartesian product of all list-valued knobs is crossed with the
`runs:` block. Each cell of that product becomes one concrete `RunSpec`.

## How it works

`expand_run_specs()` (`fltest/core/orchestrator.py`):

1. Reads the *fuzzable* knobs (`FUZZABLE_KNOBS`): `dataset`, `data_distribution`,
   `model_name`, `num_clients`, `num_rounds`, `client_epochs`, `client_lr`,
   `client_batch_size`, `dirichlet_alpha`, `classes_per_partition`, `optimizer`.
2. Treats each as a list (a scalar becomes a 1-element list).
3. Forms the cartesian product → a grid of base parameter sets.
4. Crosses the grid with `runs:` (the frameworks).
5. Resolves each cell into a flat `RunSpec` (deriving `channels`/`num_classes` from the
   dataset) with a stable `run_id`.

So the number of runs is:

```
len(runs)  ×  ∏ (length of each list-valued fuzzable knob)
```

## Example

```yaml
dataset: [mnist, cifar10]              # 2
data_distribution: [iid, dirichlet]    # 2
num_clients: [10, 20]                  # 2
runs:
  - {framework: reference}
  - {framework: flwr}                  # 2
```

→ `2 × 2 × 2 × 2 = 16` runs (8 logical configs × 2 frameworks).

## Why it matters for testing

- **Differential testing** groups runs that share the same *logical config* (everything
  except the framework) and checks they agree. Fuzzing lets you assert parity across many
  conditions at once (e.g. IID *and* non-IID, MNIST *and* CIFAR-10).
- **Pitfall checker** reads the lists directly: `data_distribution: [iid, dirichlet]` clears
  the "IID-only" pitfall; `dataset: [mnist, cifar10]` clears "MNIST-only".

## A runnable one

`examples/configs/fuzz_demo.yaml` fuzzes a single knob and crosses it with two runs:

```bash
fltest run examples/configs/fuzz_demo.yaml
```

One list (`data_distribution: [iid, dirichlet]`) crossed with two runs (undefended against
median-defended, sharing one backdoor) gives four experiments from one file. The run matrix
puts `distribution` and `defense` in columns, since those are what differ, and states the
rest once above the table.

## Preview the grid

```python
from fltest.core.config import load_config
from fltest.core.orchestrator import expand_run_specs

specs = expand_run_specs(load_config("my_conf.yaml"))
print(len(specs))
for s in specs:
    print(s.run_name, s.framework, s.dataset, s.data_distribution, s.num_clients)
```

## Notes

- Keep `dataset_partitions` fixed while sweeping `num_clients` so per-client data size stays
  comparable across the sweep (important for the `clients_scale` metamorphic relation).
- Metamorphic testing intentionally *scalarizes* lists (takes the first value) so it can
  isolate the single parameter it sweeps — see [Metamorphic testing](metamorphic-testing.md).
