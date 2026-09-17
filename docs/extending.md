# Port your attacks & defenses

This is the page for bringing **your own** technique into FLTest. Everything plugs in through
the same mechanism: subclass a base, declare the hooks you use, implement them, register a
name. Your plugin then works **across every framework backend** and composes with others —
no framework-specific code.

## The contract (read this once)

A plugin is a `HookPlugin` (`fltest/core/plugin.py`). You:

1. set `HOOKS` to the lifecycle hooks you implement;
2. implement those methods — each receives the shared `HookContext` (`ctx`);
3. register a name with a decorator so configs can reference it.

The hook lifecycle and the `ctx` fields available at each phase are in
**[Concepts & internals](concepts.md#3-the-hook-lifecycle-hookcontext)**. The two facts you
need most:

- **Parameters/updates are an ordered list of numpy arrays** (`ctx.client_update`,
  `ctx.global_state`, items in `ctx.updates_and_weights`). Operate on that list and your
  code is framework-agnostic.
- **Record metrics with `ctx.record(key=value)`** — they flow into the per-round history and
  the final result, and become assertable by the testers.

## Port an attack

Subclass `ThreatModelBaseClass`. Pick the hook by *what you attack*:

| You attack… | Hook | Mutate |
|-------------|------|--------|
| training data | `before_client_train` | `ctx.client_data` (a `{"img","label"}` loader) |
| the shared update | `after_client_train` | `ctx.client_update` (list of ndarrays) |
| via gradients (privacy) | `before_client_train` | read `ctx.global_state` + `ctx.client_data`; `ctx.record(...)` |

```python
# fltest/attacks/my_attack.py
import numpy as np
from fltest.attacks.base import ThreatModelBaseClass
from fltest.core.registry import register_attack

@register_attack("scale_update")
class ScaleUpdateAttack(ThreatModelBaseClass):
    HOOKS = ("after_client_train",)

    def __init__(self, factor: float = 5.0, **params):
        super().__init__(**params)          # handles target_clients
        self.factor = factor

    def after_client_train(self, ctx):
        if not self.targets(ctx.client_id) or ctx.client_update is None:
            return
        ctx.client_update = [u * self.factor for u in ctx.client_update]
```

Declare it in `BUILTIN_ATTACKS` in `fltest/attacks/__init__.py`, mapping the registry
name to the module that defines it:

```python
BUILTIN_ATTACKS = {
    ...
    "scale_update": "fltest.attacks.my_attack",
}
```

The module is imported the first time the attack is requested, so a plugin that pulls in
torch costs nothing until it is used. The name still has to match the one in your
`@register_attack` decorator; `tests/test_registry_and_fuzzer.py` checks that every
declared name really is registered by the module it names.

Use it:

```yaml
attacks: [{name: scale_update, params: {factor: 10.0}, target_clients: [0]}]
```

`self.targets(client_id)` returns True when the attack applies to that client
(`target_clients` is handled by the base class).

## Port a defense

Subclass `PPFLBaseClass`. Two patterns:

**Per-client perturbation** — `after_client_train`, mutate `ctx.client_update`:

```python
from fltest.core.registry import register_defense
from fltest.defenses.base import PPFLBaseClass

@register_defense("zero_small")
class ZeroSmallUpdates(PPFLBaseClass):
    HOOKS = ("after_client_train",)
    def __init__(self, threshold: float = 1e-3, **params):
        super().__init__(**params); self.threshold = threshold
    def after_client_train(self, ctx):
        if ctx.client_update is None: return
        ctx.client_update = [
            np.where(np.abs(u) < self.threshold, 0.0, u) for u in ctx.client_update
        ]
```

**Robust aggregation** — `before_aggregate`, replace the update set. Use the helpers in
`fltest/defenses/_robust.py` (`stack`, `unflatten`, `replace_with`):

```python
import numpy as np
from fltest.core.registry import register_defense
from fltest.defenses.base import PPFLBaseClass
from fltest.defenses._robust import stack, unflatten, replace_with

@register_defense("geometric_median")
class GeoMedian(PPFLBaseClass):
    HOOKS = ("before_aggregate",)
    def before_aggregate(self, ctx):
        updates = [u for u, _ in ctx.updates_and_weights]
        agg = np.median(stack(updates), axis=0)            # (replace with your rule)
        replace_with(ctx, unflatten(agg, updates[0]))      # backend averages this single result
```

Declare it in `BUILTIN_DEFENSES` in `fltest/defenses/__init__.py`
(`"geometric_median": "fltest.defenses.my_defense"`), then:

```yaml
defenses: [{name: geometric_median}]
```

For defenses that compare the same clients across rounds, `before_aggregate` also exposes
`ctx.client_submissions`: an ordered tuple of `ClientSubmission(client_id, update,
num_samples)` records. Each record corresponds to the same-position entry in
`ctx.updates_and_weights` **when the hook begins**. `ctx.global_state` contains the model
from which those clients trained. The submission records describe received updates;
continue to change `ctx.updates_and_weights` to control what is aggregated. If an earlier
hook replaces or reorders that legacy list, later hooks should not infer that its positions
still correspond to `ctx.client_submissions`. Copy arrays before retaining historical
updates, since the ndarray values are mutable. This context is available on reference and
Flower, not NVFlare. FLTest's Flower clients report stable integer IDs; if a custom Flower
client does not report one, its record has `client_id=None` and still aggregates normally.

For round-level participation, use `before_round` and assign a nonempty list or tuple of
unique client IDs to `ctx.selected_clients`. It starts with all IDs in
`range(ctx.cfg.num_clients)` on reference and Flower; setting it to `None` retains all.
Invalid IDs, duplicates, or an empty selection fail before dispatch. Flower asks clients
for their `cid` property only when selection differs from the default; custom clients
must implement `get_properties` for this use. NVFlare's replayed hook cannot select
clients.

To post-process the computed aggregate (for example, add server-side noise), implement
`on_aggregate` and assign an ordered list of model-compatible ndarrays to
`ctx.new_global_state`. Reference and Flower use that model for evaluation and subsequent
rounds, and pass it to `after_aggregate`. If the field is left unchanged or set to `None`,
the computed aggregate is used. NVFlare does not support this override.

## Port a metric

Subclass `MetricListenerBaseClass`, hook where the data you need exists, and `ctx.record(...)`:

```python
from fltest.core.registry import register_metric
from fltest.metrics.base import MetricListenerBaseClass

@register_metric("grad_norm")
class GradNormListener(MetricListenerBaseClass):
    HOOKS = ("after_client_train",)
    def after_client_train(self, ctx):
        import numpy as np
        if ctx.client_update is None: return
        norm = float(np.sqrt(sum(float((u**2).sum()) for u in ctx.client_update)))
        ctx.record(client_grad_norm=norm)
```

Declare it in `BUILTIN_METRICS` in `fltest/metrics/__init__.py`
(`"grad_norm": "fltest.metrics.my_listener"`), then use
`metrics: [accuracy, loss, grad_norm]`. The metric is now assertable in differential and
metamorphic tests (`metric: client_grad_norm`).

## Prototype without editing the package: `FLTEST_HOOKS`

For quick experiments you don't have to add files to the package. Write a hook file anywhere
and point `FLTEST_HOOKS` at it (comma-separated, no `.py`):

```python
# my_hooks.py
from fltest.core import hooks
from fltest.attacks.dlg import DLGAttack
_atk = DLGAttack(target_client=0, target_round=1)

@hooks.before_client_train
def run(ctx):
    _atk.before_client_train(ctx)
```

```bash
export FLTEST_HOOKS=my_hooks,examples/hooks/def_gradient_noise
fltest run examples/configs/dlg.yaml
```

Files are loaded in order; their hooks run in that order. (Path-loaded files run on the
driver; for the Flower backend, client-side hooks declared via the registry/config are
rebuilt inside Ray workers — see [Architecture](ARCHITECTURE.md).)

## Add a whole framework backend

Subclass `FrameworkAdapter`, implement `run_simulation(spec, data, hook_runner)`, emit the
lifecycle hooks, return a `RunResult`, and `@register_framework("name")`. Use the reference
adapter (`fltest/frameworks/reference/adapter.py`) as the template — it has the full hook
surface. Declare your module in `BUILTIN_FRAMEWORKS` in `fltest/frameworks/__init__.py`, or
in `OPTIONAL_FRAMEWORKS` if it needs a third-party dependency that may not be installed. An
optional entry is declared only when `importlib.util.find_spec` finds its dependency, and
that check does not import it.

## Test your plugin

Mirror `tests/test_attacks_defenses.py`: build a `HookContext`, call your method, assert it
mutated `ctx` as intended. Then run a real config and confirm your metric appears in the
report.
