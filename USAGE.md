# FLTest usage

See `README.md` for install and the command overview, and `docs/ARCHITECTURE.md` for design.

## Commands

| Command | What it does |
|---------|--------------|
| `fltest list` | List registered frameworks / attacks / defenses / metrics |
| `fltest run <conf>` | Run the orchestrated experiment matrix; write a JSON report |
| `fltest diff <conf>` | Differential test (cross-framework parity or determinism) |
| `fltest metamorphic <conf>` | Check metamorphic relations defined under `testing.metamorphic` |
| `fltest pitfalls <conf>` | Pitfall check + counter-experiment recommendations |

Add `-v/--verbose` to `run/diff/metamorphic` to see framework/Ray logs; `-o/--output` sets
the report directory (default `reports/`).

## Extending

- **New attack:** subclass `fltest.attacks.base.ThreatModelBaseClass`, set `HOOKS`,
  implement the methods, decorate with `@register_attack("name")`, import it in
  `fltest/attacks/__init__.py`. Usable as `attacks: [{name: name, params: {...}}]`.
- **New defense:** same pattern with `PPFLBaseClass` + `@register_defense`.
- **New metric:** `MetricListenerBaseClass` + `@register_metric`; record via `ctx.record(...)`.
- **New framework:** subclass `FrameworkAdapter`, implement `run_simulation` (emit the
  lifecycle hooks), decorate with `@register_framework("name")`.
- **New dataset/model:** add to `fltest/data/datasets.py` (`DATASET_CONFIG`/`PARTITIONERS`)
  or `fltest/data/models.py` (`MODEL_REGISTRY`).

## Runnable examples (`examples/configs/`)

| File | Demonstrates |
|------|--------------|
| `differential.yaml` | reference vs Flower parity |
| `differential_3way.yaml` | reference vs Flower vs NVFlare parity |
| `metamorphic.yaml` | clients-scale + rounds-monotonic relations |
| `attack_label_flip.yaml` | label-flip attack + per-client metric |
| `dlg.yaml` | DLG gradient-inversion privacy attack |
| `defense_robust.yaml` | backdoor defeated by median aggregation (ASR drops) |
| `pitfalls_demo.yaml` | pitfall checker on a deliberately weak setup |
