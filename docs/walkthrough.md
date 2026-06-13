# Approach walkthrough

This page traces a config from YAML to a final report, so you understand exactly what
FLTest does on your behalf.

## The pipeline

```
test_conf.yaml
   │  load_config()                         core/config.py
   ▼
TestConfig  (knobs may be scalars OR lists)
   │  expand_run_specs()                    core/orchestrator.py  ← the config fuzzer
   ▼
[ RunSpec, RunSpec, ... ]                   one flat spec per (knob-grid cell × framework run)
   │  for each spec:
   │    prepare_data(spec)                  load + partition dataset, build loaders (cached)
   │    build_hook_runner(spec)             core/wiring.py — attach metrics, attacks, defenses, FLTEST_HOOKS
   │    get_adapter(spec.framework)         core/registry.py
   ▼
adapter.run_simulation(spec, data, hook_runner)   frameworks/{reference,flower,nvflare}
   │    emits the hook lifecycle; plugins run at each phase
   ▼
RunResult  (final metrics + per-round history + extras)
   │
   ├── testing/differential.py   group runs by logical config → parity check
   ├── testing/metamorphic.py    sweep a parameter → relation check
   └── pitfalls/checker.py       inspect the config → findings + recommendations
   ▼
console summary + JSON report   (reports/<name>_*.json)
```

## Step by step

**1. Parse & validate.** `load_config()` reads the YAML into a `TestConfig` (a pydantic
model). Unknown keys are allowed; types are checked.

**2. Fuzz into runs.** `expand_run_specs()` takes the cartesian product of every
list-valued *fuzzable* knob (dataset, distribution, model, clients, rounds, lr, …) and
crosses it with the `runs:` block. Each cell becomes a fully-resolved, flat **`RunSpec`**
(channels/num_classes are derived from the dataset). See **[Config fuzzing](fuzzing.md)**.

**3. Prepare data.** `prepare_data(spec)` downloads + partitions the dataset
(IID / Dirichlet / pathological) via `flwr-datasets`, builds per-client `DataLoader`s and a
central test loader, and caches the result on disk.

**4. Wire the hooks.** `build_hook_runner(spec)` instantiates the metric listeners, attacks,
and defenses named in the config (looked up in the registries) and attaches their hooks,
plus any files in `FLTEST_HOOKS`. Attacks attach before defenses, so on a shared hook the
attack tampers first and the defense sanitizes after.

**5. Run the simulation.** The framework adapter executes federated training and **emits the
lifecycle hooks** at each phase. A single mutable `HookContext` flows through; plugins read
and mutate it. The backend records per-round centralized test loss/accuracy; attacks and
metric listeners add their own metrics.

**6. Collect results.** Each run returns a `RunResult` with `final` (last-round metrics),
`history` (per-round), and `extras`.

**7. Test & report.** Depending on the command:

   - `run` — just prints/saves the matrix.
   - `diff` — groups runs that differ only by framework and checks metric **parity**.
   - `metamorphic` — sweeps a parameter and checks a **relation**.
   - `pitfalls` — inspects the config and prints findings + counter-experiments.

## What flows through a run: the hook lifecycle

```
before_simulation
on_data_distribute
for each round:
    before_round
    for each client:
        before_client_train      # attacks poison data / DLG reconstructs
        (local training)
        after_client_train        # attacks tamper update / defenses clip+noise
    before_aggregate              # robust-aggregation defenses replace the update set
    on_aggregate / after_aggregate
    after_round                   # evaluate global model; metric listeners record
after_simulation                  # personalized metrics; final accuracy
```

Continue to **[Concepts & internals](concepts.md)** for the data structures, or jump to
**[Configuration reference](configuration.md)** to start building configs.
