# Concepts & internals

The five ideas that make FLTest work.

## 1. `RunSpec` — one flat, resolved run

A `RunSpec` (`fltest/core/config.py`) is everything one simulation needs: framework, seed,
device, dataset + distribution, model, FL params (clients/rounds/lr/epochs/batch), the
attacks/defenses/metrics to compose, and cache paths. The fuzzer produces a list of these
from your `TestConfig`. Adapters consume only `RunSpec` — never the raw YAML.

## 2. The `FrameworkAdapter` seam

Every FL framework implements one method (`fltest/frameworks/base.py`):

```python
class FrameworkAdapter:
    def run_simulation(self, spec: RunSpec, data: dict, hook_runner: HookRunner) -> RunResult: ...
```

The core never imports a framework directly; `get_adapter(name)` returns the registered
adapter. This is the "`run_simulation()` boundary": adding a backend is a subclass +
`@register_framework("name")`. See **[Architecture](ARCHITECTURE.md)** for the fidelity
table (which hooks each backend supports).

## 3. The hook lifecycle + `HookContext`

A single mutable `HookContext` (`fltest/core/hook_context.py`) flows through every phase:

```
before_simulation → on_data_distribute →
  [ before_round →
    { before_client_train → (train) → after_client_train }* →
    before_aggregate → on_aggregate → after_aggregate → after_round ]* →
after_simulation
```

Relevant `HookContext` fields by phase:

| Field | Set during | Meaning |
|-------|-----------|---------|
| `cfg`, `framework`, `round`, `client_id` | all | run identity |
| `dist_dict` | data phase | `cid → client loader` (attacks may repartition) |
| `client_data` | `before_client_train` | this client's loader (attacks swap it) |
| `global_state` | rounds | current global params (list of ndarrays) |
| `client_update` | `after_client_train` | this client's update (attacks/defenses mutate) |
| `updates_and_weights` | `before_aggregate` | `[(update, n), …]` (robust agg replaces this) |
| `model`, `test_data` | `after_round` | live model + central test loader |
| `metrics`, `history` | all | `ctx.record(**kv)` writes here |

## 4. Hook plugins (`HookPlugin`)

Attacks, defenses, and metric listeners all subclass `HookPlugin`
(`fltest/core/plugin.py`). A plugin declares the hooks it uses in `HOOKS` and implements
the matching methods; `attach()` registers exactly those onto a `HookRunner`.

```python
class MyAttack(ThreatModelBaseClass):
    HOOKS = ("before_client_train",)
    def before_client_train(self, ctx): ...
```

Because every backend emits the same hooks with the same context, a plugin written once
runs across all of them, and multiple plugins compose on one run. See
**[Port your attacks & defenses](extending.md)**.

## 5. One canonical parameter representation

Model parameters and updates are always an **ordered list of numpy arrays**
(`state_dict_to_ndarrays` / `load_ndarrays_into` / `aggregate_ndarrays` in
`fltest/data/utils.py`). Attacks and defenses operate on this list, so they are identical
across frameworks regardless of each framework's native format. The reference backend and
Flower both aggregate with the same FLTest weighted-mean, which is what makes their results
match within tolerance.

## `RunResult`

```python
@dataclass
class RunResult:
    framework: str
    status: str                 # "success" | "failed"
    final: dict                 # last-round metrics, e.g. {"accuracy":.., "loss":..}
    history: dict[int, dict]    # round → metrics
    per_client: dict            # optional
    extras: dict                # e.g. DLG reconstruction details
    duration_seconds: float
```

`final` is what differential and metamorphic tests read.
