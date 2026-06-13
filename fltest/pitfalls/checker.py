"""Pitfall checker: heuristic detectors for the six FL-evaluation pitfalls.

Maps directly onto the pitfalls catalogued in the proposal (Section 3.1). Each detector
inspects a :class:`TestConfig` (the planned evaluation) and emits a :class:`Finding` when
the setup risks over-estimating privacy/robustness. The recommender turns findings into
concrete counter-experiments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from fltest.core.config import TestConfig

# Attacks the proposal flags as "naive" (Pitfall-1): over-used, weak under strong settings.
NAIVE_ATTACKS = {"gaussian", "label_flip", "sign_flip"}
# Class-balanced datasets that don't represent real-world heterogeneity (Pitfall-2).
BALANCED_DATASETS = {"mnist", "fashion_mnist", "cifar10"}
PRIVACY_ATTACKS = {"dlg"}  # gradient-inversion / inference style


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
            "Add at least one strong attack (e.g. backdoor) and one privacy attack (dlg).",
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
            "Add a harder/real-world dataset and non-IID partitioning.",
            {"datasets": sorted(datasets)}))
    elif datasets and datasets <= BALANCED_DATASETS:
        findings.append(Finding(
            "P2_dataset", "Class-balanced datasets only", "low",
            "All datasets are class-balanced; they don't reflect real-world label imbalance.",
            "Include a naturally non-IID/real-world dataset where available.",
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
