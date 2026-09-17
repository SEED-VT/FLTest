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
    # femnist is writer-partitioned, so it clears P2 outright rather than downgrading it.
    "P2_dataset": "dataset: [mnist, cifar100, femnist]\ndata_distribution: [iid, dirichlet]",
    "P3_iid_only": "data_distribution: [iid, dirichlet, pathological]",
    "P3_no_personalized": "metrics: [accuracy, loss, per_client]",
    "P4_misconfig_dp": (
        "# sweep DP noise to chart the privacy/utility trade-off\n"
        "defenses:\n  - {name: gradient_noise, params: {clip_norm: 1.0, sigma: 0.05}}\n"
        "testing:\n  metamorphic:\n    - {relation: dp_noise, parameter: defense.sigma, "
        "values: [0.0, 0.05, 0.1, 0.2], metric: accuracy}"
    ),
    # Membership inference comes first: it is cheap, it scores every round, and unlike
    # gradient inversion it applies to text as well as images.
    "P5_subtle_leakage": (
        "attacks:\n"
        "  - {name: membership_inference, params: {target_client: 0}}\n"
        "  - {name: dlg, params: {target_round: 1, iters: 300}}"
    ),
    "P4_misconfig_secagg": (
        "# give the ring enough headroom for sample-weighted updates, and no dropouts\n"
        "defenses:\n"
        "  - {name: mpc_aggregation, params: {quant_bits: 16, modulus: 4294967296, "
        "dropout_rate: 0.0}}\n"
        "# then confirm the arithmetic: mpc_agg_max_abs_error should sit at the "
        "quantization floor"
    ),
    "P4_untested_secagg": (
        "# the masks either cancel exactly or they do not - check it with an equality oracle\n"
        "testing:\n  metamorphic:\n    - {relation: secagg_lossless, parameter: defense.seed, "
        "values: [1, 2, 3], metric: gm_weight_sum, tolerance: 0.0}"
    ),
    "P4_secagg_vs_robust": (
        "# split the stack: a masked server cannot inspect updates client-by-client\n"
        "runs:\n"
        "  - {framework: reference, name: secagg, defenses: [{name: secure_aggregation}]}\n"
        "  - {framework: reference, name: robust, defenses: [{name: median}]}"
    ),
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
