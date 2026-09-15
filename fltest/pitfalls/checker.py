"""Pitfall checker: heuristic detectors for the six FL-evaluation pitfalls.

Maps directly onto the pitfalls catalogued in the proposal (Section 3.1). Each detector
inspects a :class:`TestConfig` (the planned evaluation) and emits a :class:`Finding` when
the setup risks over-estimating privacy/robustness. The recommender turns findings into
concrete counter-experiments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from fltest.core.config import ROBUST_AGGREGATORS, TestConfig

# Attacks the proposal flags as "naive" (Pitfall-1): over-used, weak under strong settings.
NAIVE_ATTACKS = {"gaussian", "label_flip", "sign_flip"}
# Class-balanced datasets that don't represent real-world heterogeneity (Pitfall-2).
# FEMNIST is deliberately absent: it is writer-partitioned and naturally non-IID, so a
# config that uses it clears this pitfall.
BALANCED_DATASETS = {"mnist", "fashion_mnist", "cifar10", "cifar100"}
PRIVACY_ATTACKS = {"dlg", "membership_inference"}  # gradient-inversion / inference style


@dataclass
class Finding:
    pitfall: str           # short id, e.g. "P1_threat_models"
    title: str
    severity: str          # "high" | "medium" | "low"
    message: str
    recommendation: str
    evidence: Dict[str, Any] = field(default_factory=dict)


def _as_list(v):
    return v if isinstance(v, list) else [v]


def check_config(config: TestConfig) -> List[Finding]:
    findings: List[Finding] = []

    # Aggregate over both the top-level config and any per-run overrides, so matrix configs
    # that declare attacks/defenses/metrics/datasets inside `runs:` are assessed correctly.
    datasets = set(_as_list(config.dataset))
    distributions = set(_as_list(config.data_distribution))
    attack_names = {a.name for a in config.attacks}
    defense_names = {d.name for d in config.defenses}
    metrics = set(config.metrics)

    for run in config.runs:
        datasets |= set(_as_list(run.get("dataset", []))) if run.get("dataset") else set()
        distributions |= set(_as_list(run.get("data_distribution", []))) if run.get("data_distribution") else set()
        attack_names |= {a["name"] for a in run.get("attacks", []) if isinstance(a, dict) and "name" in a}
        defense_names |= {d["name"] for d in run.get("defenses", []) if isinstance(d, dict) and "name" in d}
        metrics |= set(run.get("metrics", []))

    # P1 — Inadequate testing against relevant threat models.
    if not attack_names:
        findings.append(Finding(
            "P1_threat_models", "Inadequate threat models", "high",
            "No attacks configured; robustness/privacy claims would be untested.",
            "Add a strong attack such as 'backdoor', and a privacy attack such as "
            "'membership_inference' or 'dlg'.",
            {"attacks": sorted(attack_names)}))
    elif attack_names and attack_names <= NAIVE_ATTACKS:
        findings.append(Finding(
            "P1_threat_models", "Only naive attacks", "medium",
            f"Only naive attacks used ({sorted(attack_names)}); ~40% of works rely on these "
            "and they are weak under strong/adaptive settings.",
            "Add a stronger attack such as 'backdoor' and a privacy attack 'dlg'.",
            {"attacks": sorted(attack_names)}))

    # P2 — Overlooking dataset sensitivities.
    if datasets and datasets <= {"mnist"}:
        findings.append(Finding(
            "P2_dataset", "MNIST-only evaluation", "medium",
            "Only MNIST is used; it under-represents real-world complexity/heterogeneity.",
            "Add a harder dataset such as 'cifar100', or 'femnist' for real-world skew.",
            {"datasets": sorted(datasets)}))
    elif datasets and datasets <= BALANCED_DATASETS:
        findings.append(Finding(
            "P2_dataset", "Class-balanced datasets only", "low",
            "All datasets are class-balanced; they don't reflect real-world label imbalance.",
            "Add 'femnist', which is partitioned by writer and naturally non-IID.",
            {"datasets": sorted(datasets)}))

    # P3 — Standardization of procedures & metrics (IID-only + no personalized eval).
    if distributions and distributions <= {"iid"}:
        findings.append(Finding(
            "P3_iid_only", "IID-only data distribution", "high",
            "Only IID is evaluated; ~50% of works do this even though IID is easiest to defend.",
            "Sweep data_distribution over ['iid','dirichlet','pathological'].",
            {"distributions": sorted(distributions)}))
    if "per_client" not in metrics:
        findings.append(Finding(
            "P3_no_personalized", "No personalized evaluation", "medium",
            "Only global metrics tracked; per-client (personalized) accuracy is rarely reported "
            "(~4% of works) yet reveals representation disparity.",
            "Add 'per_client' to metrics.",
            {"metrics": sorted(metrics)}))

    # P4 — Misconfiguration of privacy-preserving techniques.
    for d in config.defenses:
        if d.name == "gradient_noise":
            sigma = d.params.get("sigma", 0.01)
            if sigma == 0:
                findings.append(Finding(
                    "P4_misconfig_dp", "DP noise disabled", "high",
                    "gradient_noise sigma=0 gives no privacy while appearing to use DP.",
                    "Set a positive sigma and sweep it to study the privacy/utility trade-off.",
                    {"sigma": sigma}))
            elif sigma >= 1.0:
                findings.append(Finding(
                    "P4_misconfig_dp", "DP noise likely too large", "low",
                    f"gradient_noise sigma={sigma} may destroy utility.",
                    "Sweep sigma to find a usable privacy/utility operating point.",
                    {"sigma": sigma}))

    # P4 (continued) — misconfiguration of the secure-aggregation techniques. Their failure
    # modes are silent: a run with masking effectively disabled, or with a ring too small for
    # the values it carries, still produces a finite, plausible-looking aggregate.
    for d in config.defenses:
        if d.name == "secure_aggregation":
            mask_scale = d.params.get("mask_scale", 1.0)
            if mask_scale <= 0:
                findings.append(Finding(
                    "P4_misconfig_secagg", "Secure aggregation masks disabled", "high",
                    f"secure_aggregation mask_scale={mask_scale} adds no mask, so the server "
                    "sees every update in the clear while the config claims SecAgg.",
                    "Set a positive mask_scale, and read secagg_mask_to_update_ratio from the "
                    "run to confirm the mask actually dominates the update.",
                    {"mask_scale": mask_scale}))
        if d.name == "mpc_aggregation":
            quant_bits = d.params.get("quant_bits", 16)
            modulus = d.params.get("modulus", 1 << 32)
            dropout_rate = d.params.get("dropout_rate", 0.0)
            if dropout_rate > 0:
                findings.append(Finding(
                    "P4_misconfig_secagg", "MPC dropouts without recovery", "high",
                    f"mpc_aggregation dropout_rate={dropout_rate} drops clients after they "
                    "mask. FLTest does not implement threshold secret sharing, so the dropped "
                    "clients' pairwise masks stay in the sum and the aggregate is wrong.",
                    "Keep dropout_rate=0 for utility runs; use a positive value only to "
                    "measure the failure mode via mpc_agg_max_abs_error.",
                    {"dropout_rate": dropout_rate}))
            if quant_bits < 8:
                findings.append(Finding(
                    "P4_misconfig_secagg", "Fixed-point precision likely too low", "medium",
                    f"mpc_aggregation quant_bits={quant_bits} leaves under 8 fractional bits; "
                    "quantization error can exceed the update it is meant to carry.",
                    "Raise quant_bits and sweep it against mpc_agg_max_abs_error to find the "
                    "precision floor.",
                    {"quant_bits": quant_bits}))
            # The ring carries sum_i n_i * x_i scaled by 2**quant_bits, and holds it only
            # while |value| < modulus/2. Individual shares may wrap harmlessly; the summed
            # aggregate leaving that band is what cannot be recovered.
            headroom = modulus / 2 / (2 ** quant_bits)
            if headroom < 1.0:
                findings.append(Finding(
                    "P4_misconfig_secagg", "MPC ring too small for its precision", "high",
                    f"mpc_aggregation can represent |value| < {headroom:.3g} with "
                    f"quant_bits={quant_bits}, modulus={modulus}, but it must hold the "
                    "sample-weighted sum of every client update. Overflow wraps around "
                    "silently rather than raising.",
                    "Raise modulus or lower quant_bits, and check mpc_overflow_rate is 0.",
                    {"quant_bits": quant_bits, "modulus": modulus, "headroom": headroom}))

    # P4 (continued) — a secure-aggregation run with no exact-equality oracle. Masking is
    # either lossless or broken, and accuracy thresholds are too coarse to tell them apart.
    secagg_names = {"secure_aggregation", "mpc_aggregation"} & defense_names
    if secagg_names:
        relations = {r.relation for r in config.testing.metamorphic}
        if "secagg_lossless" not in relations:
            findings.append(Finding(
                "P4_untested_secagg", "Secure aggregation correctness untested", "medium",
                f"{sorted(secagg_names)} is configured but no 'secagg_lossless' relation "
                "checks that the masks cancel. A residue that survives aggregation moves the "
                "model without necessarily moving accuracy far enough to notice.",
                "Add a secagg_lossless metamorphic relation over defense.seed with "
                "tolerance 0 on the gm_weight_sum metric.",
                {"defenses": sorted(secagg_names), "relations": sorted(relations)}))

    # P4 (continued) — masking and per-client robust aggregation are mutually exclusive in a
    # real deployment. Both "work" here because the simulation hands the server plaintext, so
    # a config combining them measures a system nobody can build.
    if "secure_aggregation" in defense_names and (defense_names & set(ROBUST_AGGREGATORS)):
        findings.append(Finding(
            "P4_secagg_vs_robust", "Secure aggregation combined with robust aggregation", "medium",
            "secure_aggregation hides individual updates, but "
            f"{sorted(defense_names & set(ROBUST_AGGREGATORS))} needs to compare them "
            "client-by-client. A real server cannot do both; this simulation lets it, so the "
            "combination over-states what the defense stack delivers.",
            "Evaluate the two separately, or state explicitly that the robust rule assumes an "
            "unmasked server.",
            {"defenses": sorted(defense_names)}))

    # P5 — Underestimating subtle privacy leakages.
    if not (attack_names & PRIVACY_ATTACKS):
        findings.append(Finding(
            "P5_subtle_leakage", "No privacy-leakage attack", "medium",
            "No gradient-inversion/inference attack (e.g. dlg) is included, so subtle leakage "
            "through shared updates goes untested.",
            "Add the 'dlg' attack and measure reconstruction quality.",
            {"attacks": sorted(attack_names)}))

    # P6 — Overestimating user expertise (DP present but no robust-aggregation safety net).
    if defense_names and defense_names <= {"gradient_noise", "norm_clip"} and (attack_names - NAIVE_ATTACKS):
        findings.append(Finding(
            "P6_user_expertise", "Possibly mismatched defense for threat", "low",
            "Only perturbation defenses are configured against non-naive attacks; robust "
            "aggregation may be needed.",
            "Consider adding a robust-aggregation defense (krum / trimmed_mean / median).",
            {"defenses": sorted(defense_names), "attacks": sorted(attack_names)}))

    return findings
