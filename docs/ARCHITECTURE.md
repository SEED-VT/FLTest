# FLTest architecture

```
            test_conf.yaml
                  │
                  ▼
        core/config.py  (TestConfig)              core/registry.py
                  │                          (frameworks / attacks /
                  ▼                            defenses / metrics)
      core/orchestrator.py                              │
   ── config fuzzer: lists → grid ──┐                   │
   ── × runs (frameworks) ──────────┘                   │
                  │ RunSpec (flat, per run)              │
                  ▼                                      ▼
        core/wiring.build_hook_runner(spec) ── attaches ── attacks/  defenses/  metrics/
                  │  (HookRunner with all plugin hooks)        (HookPlugin subclasses)
                  ▼
      frameworks/base.FrameworkAdapter.run_simulation(spec, data, hook_runner)
        ├─ reference/  (pure PyTorch FedAvg — oracle, full hook surface)
        ├─ flower/     (flwr simulation; client-side hooks rebuilt in Ray workers)
        └─ nvflare/    (NVFlare simulator; driver-side hooks only)   [optional extra]
                  │ RunResult (final metrics, per-round history)
                  ▼
        testing/ (differential, metamorphic)   pitfalls/ (checker + recommender)
                  │
                  ▼
        testing/report.py  → console + JSON
```

## The seam: `run_simulation()`

`frameworks/base.py` defines `FrameworkAdapter.run_simulation(spec, data, hook_runner) ->
RunResult`. The core never imports a framework directly; `get_adapter(name)` returns the
registered adapter. Adding a backend = subclass + `@register_framework("name")` + emit the
lifecycle hooks. `frameworks/__init__.py` registers reference + Flower always and tries the
optional heavy backends (NVFlare) only if importable.

## The hook lifecycle

`before_simulation → on_data_distribute → [ before_round →
{ before_client_train → (train) → after_client_train }* →
before_aggregate → on_aggregate → after_aggregate → after_round ]* → after_simulation`

A single mutable `HookContext` (core/hook_context.py) flows through these. Plugins are
`HookPlugin` subclasses (core/plugin.py): they declare the hooks they use in `HOOKS` and
implement matching methods; `attach()` registers exactly those onto a `HookRunner`.

- **Attacks** (`attacks/`, `ThreatModelBaseClass`): mutate `ctx.client_data` (data poisoning),
  `ctx.client_update` (model poisoning), or reconstruct from gradients (DLG).
- **Defenses** (`defenses/`, `PPFLBaseClass`): perturb `ctx.client_update` per client
  (gradient_noise, norm_clip) or replace `ctx.updates_and_weights` at `before_aggregate`
  (robust aggregation: krum, trimmed_mean, median).
- **Metric listeners** (`metrics/`): record extra metrics via `ctx.record(...)`.

Parameters use one canonical representation everywhere — an ordered list of numpy arrays —
so plugins are framework-agnostic. Registration order means attacks attach before defenses,
so a defense sanitizes a tampered update on the same hook.

## Backend hook fidelity

| Backend   | data hooks | client-side hooks | server/round hooks | notes |
|-----------|:---------:|:-----------------:|:------------------:|-------|
| reference | ✓ | ✓ | ✓ | the oracle; develop attacks here |
| flower    | ✓ | ✓ (rebuilt in Ray workers from the spec) | ✓ | aggregates with FLTest's own weighted mean for parity |
| nvflare   | ✓ (driver) | — (separate processes) | ✓ (replayed snapshots) | differential parity of vanilla FedAvg |

## Testing engines

- **Differential** (`testing/differential.py`): groups runs by a *logical key* (all spec
  fields except the framework) and asserts the chosen metric is within tolerance across the
  group (cross-framework parity); or re-runs a spec and asserts identical results
  (determinism). Rules in `testing/rules.py`.
- **Metamorphic** (`testing/metamorphic.py`): transforms one input parameter over a sweep
  and checks a relation (`non_decreasing` / `non_increasing`) — e.g. N→2N clients must not
  drop accuracy; stronger attack must not raise it.

## Pitfalls

`pitfalls/checker.py` maps the proposal's six pitfalls to heuristic detectors over a
`TestConfig`; `pitfalls/recommender.py` turns findings into counter-experiment YAML snippets.
