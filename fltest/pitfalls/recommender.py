"""Counter-experiment recommender.

Turns pitfall findings into concrete, copy-pasteable config adjustments — the proposal's
recommendation engine that suggests minimal changes to strengthen an evaluation.
"""

from __future__ import annotations

from typing import Dict, List

from fltest.pitfalls.checker import Finding

# Counter-experiment snippets keyed by pitfall id (YAML fragments the user can merge).
_COUNTER_EXPERIMENTS: Dict[str, str] = {
    "P1_threat_models": (
        "attacks:\n"
        "  - {name: backdoor, params: {target_label: 0, infection_rate: 0.3}}\n"
        "  - {name: dlg, params: {target_round: 1}}"
    ),
    "P2_dataset": "dataset: [mnist, cifar10]\ndata_distribution: [iid, dirichlet]",
    "P3_iid_only": "data_distribution: [iid, dirichlet, pathological]",
    "P3_no_personalized": "metrics: [accuracy, loss, per_client]",
    "P4_misconfig_dp": (
        "# sweep DP noise to chart the privacy/utility trade-off\n"
        "defenses:\n  - {name: gradient_noise, params: {clip_norm: 1.0, sigma: 0.05}}\n"
        "testing:\n  metamorphic:\n    - {relation: dp_noise, parameter: defense.sigma, "
        "values: [0.0, 0.05, 0.1, 0.2], metric: accuracy}"
    ),
    "P5_subtle_leakage": "attacks:\n  - {name: dlg, params: {target_round: 1, iters: 300}}",
    "P6_user_expertise": "defenses:\n  - {name: krum, params: {num_byzantine: 1}}",
}


def recommend(findings: List[Finding]) -> List[Dict[str, str]]:
    """Return ordered recommendations (highest severity first) with counter-experiments."""
    order = {"high": 0, "medium": 1, "low": 2}
    out: List[Dict[str, str]] = []
    seen = set()
    for f in sorted(findings, key=lambda x: order.get(x.severity, 3)):
        if f.pitfall in seen:
            continue
        seen.add(f.pitfall)
        out.append({
            "pitfall": f.pitfall,
            "title": f.title,
            "severity": f.severity,
            "recommendation": f.recommendation,
            "counter_experiment": _COUNTER_EXPERIMENTS.get(f.pitfall, ""),
        })
    return out
