"""Metamorphic testing for FL (slide 22).

A metamorphic relation transforms the input config along one parameter and asserts an
expected relation on the output metric — no ground-truth label needed. Built-in relations:

* ``clients_scale``    — vary ``num_clients`` (e.g. N, 2N); accuracy must not drop (IID).
* ``rounds_monotonic`` — vary ``num_rounds``; accuracy must be non-decreasing.
* ``attack_strength``  — vary an attack parameter (dotted, e.g. ``attack.infection_rate``);
                          accuracy must be non-increasing as the attack strengthens.
* ``dp_noise``         — vary a defense parameter (e.g. ``defense.sigma``);
                          accuracy must be non-increasing as privacy noise increases.

Metamorphic checks are within a single framework (the first entry in ``runs``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from fltest.core.config import FUZZABLE_KNOBS, MetamorphicRelation, TestConfig
from fltest.core.orchestrator import Orchestrator, expand_run_specs
from fltest.testing.report import TestOutcome
from fltest.testing.rules import non_decreasing, non_increasing

# relation -> (default parameter, rule). ``None`` parameter must be supplied by the user.
_RELATIONS = {
    "clients_scale": ("num_clients", non_decreasing),
    "rounds_monotonic": ("num_rounds", non_decreasing),
    "attack_strength": (None, non_increasing),
    "dp_noise": (None, non_increasing),
}


def _scalarize(config: TestConfig) -> Dict[str, Any]:
    """Collapse any list-valued fuzzable knob to its first value (isolate the relation)."""
    d = config.model_dump()
    for knob in FUZZABLE_KNOBS:
        if isinstance(d.get(knob), list):
            d[knob] = d[knob][0]
    d["runs"] = [config.runs[0]]  # single framework for metamorphic checks
    return d


def _set_param(config_dict: Dict[str, Any], param: str, value: Any) -> None:
    """Set ``param`` (top-level, or dotted ``attack.x`` / ``defense.x``) to ``value``."""
    if param.startswith("attack.") or param.startswith("defense."):
        kind, sub = param.split(".", 1)
        key = "attacks" if kind == "attack" else "defenses"
        specs = config_dict.get(key) or []
        if not specs:
            raise ValueError(f"metamorphic relation needs at least one {kind} to vary '{param}'")
        specs[0].setdefault("params", {})[sub] = value
    else:
        config_dict[param] = value


@dataclass
class MetamorphicReport:
    outcomes: List[TestOutcome] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(o.status != "FAIL" for o in self.outcomes)


class MetamorphicTester:
    def __init__(self, orchestrator: Optional[Orchestrator] = None):
        self.orchestrator = orchestrator or Orchestrator()

    def _run_value(self, base: Dict[str, Any], param: str, value: Any) -> Optional[float]:
        cfg_dict = {**base, "attacks": [dict(a) for a in base.get("attacks", [])],
                    "defenses": [dict(d) for d in base.get("defenses", [])]}
        _set_param(cfg_dict, param, value)
        cfg = TestConfig(**cfg_dict)
        spec = expand_run_specs(cfg)[0]
        result = self.orchestrator.run_spec(spec)
        if result.status != "success":
            return None
        return result.final.get(self._metric)

    def check(self, config: TestConfig, relations: List[MetamorphicRelation]) -> MetamorphicReport:
        report = MetamorphicReport()
        base = _scalarize(config)

        for rel in relations:
            if rel.relation not in _RELATIONS:
                report.outcomes.append(TestOutcome(
                    "metamorphic", rel.relation, "SKIP", f"unknown relation '{rel.relation}'"))
                continue
            default_param, rule = _RELATIONS[rel.relation]
            param = rel.parameter or default_param
            if param is None:
                report.outcomes.append(TestOutcome(
                    "metamorphic", rel.relation, "SKIP", "relation requires an explicit 'parameter'"))
                continue
            values = rel.values
            if not values:
                report.outcomes.append(TestOutcome(
                    "metamorphic", rel.relation, "SKIP", "relation requires 'values' to sweep"))
                continue

            self._metric = rel.metric
            pairs = []
            for v in values:
                metric_val = self._run_value(base, param, v)
                if metric_val is None:
                    report.outcomes.append(TestOutcome(
                        "metamorphic", f"{rel.relation} ({param})", "FAIL",
                        f"run failed at {param}={v}"))
                    pairs = None
                    break
                pairs.append((float(v) if isinstance(v, (int, float)) else len(pairs), metric_val))
            if pairs is None:
                continue

            ok, detail = rule(pairs, rel.tolerance)
            report.outcomes.append(TestOutcome(
                "metamorphic", f"{rel.relation} ({param}, metric={rel.metric})",
                "PASS" if ok else "FAIL", detail,
                evidence={"param": param, "values": values,
                          rel.metric: [round(p[1], 4) for p in pairs]}))
        return report
