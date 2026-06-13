# Pitfall checker

The pitfall checker inspects a config (before or without running it) and flags the common
FL-evaluation mistakes from the proposal, then recommends concrete counter-experiments.
Code: `fltest/pitfalls/`.

```bash
fltest pitfalls <config.yaml>
```

## Detectors

| Id | Pitfall | Triggers when… | Severity |
|----|---------|----------------|----------|
| `P1_threat_models` | Inadequate threat models | no attacks, or only naive ones (`gaussian`/`label_flip`/`sign_flip`) | high / medium |
| `P2_dataset` | Dataset sensitivities | MNIST-only, or only class-balanced datasets | medium / low |
| `P3_iid_only` | IID-only evaluation | every `data_distribution` is `iid` | high |
| `P3_no_personalized` | No personalized metric | `per_client` not in `metrics` | medium |
| `P4_misconfig_dp` | Misconfigured DP | `gradient_noise` with `sigma=0` (no privacy) or very large | high / low |
| `P5_subtle_leakage` | Subtle privacy leakage | no privacy attack (`dlg`) included | medium |
| `P6_user_expertise` | Mismatched defense | only perturbation defenses against non-naive attacks | low |

## Recommendations

For each finding the recommender emits a copy-pasteable YAML fragment, ordered by severity.
Example output:

```
[HIGH  ] IID-only data distribution  (P3_iid_only)
    Only IID is evaluated; ~50% of works do this even though IID is easiest to defend.
    → Sweep data_distribution over ['iid','dirichlet','pathological'].

# IID-only data distribution (high)
data_distribution: [iid, dirichlet, pathological]
```

Merge the fragment and the pitfall clears on the next check (e.g. the list form satisfies
the [fuzzer](fuzzing.md) and exercises non-IID).

## Programmatic use

```python
from fltest.core.config import load_config
from fltest.pitfalls import check_config, recommend

cfg = load_config("my_conf.yaml")
findings = check_config(cfg)
for f in findings:
    print(f.severity, f.pitfall, f.message)
for r in recommend(findings):
    print(r["title"], r["counter_experiment"])
```

Add or tune detectors in `fltest/pitfalls/checker.py` and their counter-experiments in
`fltest/pitfalls/recommender.py`.
