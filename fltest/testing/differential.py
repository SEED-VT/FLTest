"""Differential testing for FL frameworks and PPFL techniques (slide 21).

Two modes:

* **cross_framework** — runs that differ *only* by framework should reach the same final
  metric within tolerance. Surfaces framework-implementation divergences (the kind that
  found the AppFL CUDA-on-CPU bug).
* **determinism** — running the same spec twice must give identical results. Surfaces
  hidden nondeterminism / state leakage.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from fltest.core.config import RunSpec, TestConfig
from fltest.core.orchestrator import Orchestrator, RunMatrix
from fltest.frameworks.base import RunResult
from fltest.testing.report import TestOutcome
from fltest.testing.rules import parity

# Spec fields that define the *logical* experiment (everything but the backend identity).
_LOGICAL_FIELDS = (
    "dataset", "data_distribution", "dirichlet_alpha", "classes_per_partition",
    "model_name", "num_clients", "num_rounds", "client_epochs", "client_lr",
    "client_batch_size", "optimizer", "seed",
)


def logical_key(spec: RunSpec) -> str:
    payload: Dict[str, Any] = {f: getattr(spec, f) for f in _LOGICAL_FIELDS}
    payload["attacks"] = [(a.name, a.params, a.target_clients) for a in spec.attacks]
    payload["defenses"] = [(d.name, d.params) for d in spec.defenses]
    return json.dumps(payload, sort_keys=True, default=str)


@dataclass
class DifferentialReport:
    mode: str
    metric: str
    tolerance: float
    outcomes: List[TestOutcome] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(o.status != "FAIL" for o in self.outcomes)


class DifferentialTester:
    def __init__(self, orchestrator: Optional[Orchestrator] = None):
        self.orchestrator = orchestrator or Orchestrator()

    def cross_framework(self, matrix: RunMatrix, metric: str = "accuracy", tolerance: float = 0.05) -> DifferentialReport:
        report = DifferentialReport(mode="cross_framework", metric=metric, tolerance=tolerance)

        groups: Dict[str, List[tuple]] = {}
        for spec, result in zip(matrix.specs, matrix.results):
            groups.setdefault(logical_key(spec), []).append((spec, result))

        for key, members in groups.items():
            frameworks = [s.framework for s, _ in members]
            sample = members[0][0]
            label = f"{sample.dataset}/{sample.data_distribution} {sample.model_name} " \
                    f"c{sample.num_clients} r{sample.num_rounds} :: {frameworks}"
            if len(members) < 2:
                report.outcomes.append(TestOutcome(
                    "differential", label, "SKIP", "need >=2 frameworks on the same config"))
                continue
            failed = [r for _, r in members if r.status != "success"]
            if failed:
                report.outcomes.append(TestOutcome(
                    "differential", label, "FAIL",
                    f"{len(failed)} run(s) errored: {[r.framework for r in failed]}"))
                continue
            vals = [r.final.get(metric) for _, r in members]
            if any(v is None for v in vals):
                report.outcomes.append(TestOutcome(
                    "differential", label, "SKIP", f"metric '{metric}' missing in some runs"))
                continue
            ok, detail = parity(vals, tolerance)
            report.outcomes.append(TestOutcome(
                "differential", label, "PASS" if ok else "FAIL", detail,
                evidence={"frameworks": frameworks, metric: [round(v, 4) for v in vals]}))
        return report

    def determinism(self, config: TestConfig, metric: str = "accuracy", eps: float = 1e-4) -> DifferentialReport:
        from fltest.core.orchestrator import expand_run_specs

        report = DifferentialReport(mode="determinism", metric=metric, tolerance=eps)
        for spec in expand_run_specs(config):
            r1 = self.orchestrator.run_spec(spec)
            r2 = self.orchestrator.run_spec(spec)
            label = f"{spec.run_name} [{spec.framework}] {spec.dataset} c{spec.num_clients} r{spec.num_rounds}"
            if r1.status != "success" or r2.status != "success":
                report.outcomes.append(TestOutcome("differential", label, "FAIL", "a repeat run errored"))
                continue
            v1, v2 = r1.final.get(metric), r2.final.get(metric)
            if v1 is None or v2 is None:
                report.outcomes.append(TestOutcome("differential", label, "SKIP", f"metric '{metric}' missing"))
                continue
            ok = abs(v1 - v2) <= eps
            report.outcomes.append(TestOutcome(
                "differential", label, "PASS" if ok else "FAIL",
                f"run1={v1:.6f} run2={v2:.6f} |Δ|={abs(v1 - v2):.2e} (eps={eps})"))
        return report
