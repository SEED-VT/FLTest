"""Reporting for runs and test outcomes (JSON file + console summary)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class TestOutcome:
    """A single PASS/FAIL check produced by a tester."""

    test_type: str          # "differential" | "metamorphic"
    name: str               # what was checked
    status: str             # "PASS" | "FAIL" | "SKIP"
    detail: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)


def summarize(outcomes: List[TestOutcome]) -> Dict[str, int]:
    counts = {"PASS": 0, "FAIL": 0, "SKIP": 0}
    for o in outcomes:
        counts[o.status] = counts.get(o.status, 0) + 1
    return counts


def write_report(
    path: str | Path,
    title: str,
    runs: Optional[List[Dict[str, Any]]] = None,
    outcomes: Optional[List[TestOutcome]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write a JSON report and return its path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "title": title,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "runs": runs or [],
        "outcomes": [asdict(o) for o in (outcomes or [])],
        "summary": summarize(outcomes or []),
    }
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def print_outcomes(title: str, outcomes: List[TestOutcome]) -> bool:
    """Print a console table of outcomes. Returns True if all passed (no FAILs)."""
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")
    symbol = {"PASS": "✓", "FAIL": "✗", "SKIP": "–"}
    for o in outcomes:
        print(f"  [{symbol.get(o.status, '?')} {o.status}] {o.test_type}: {o.name}")
        if o.detail:
            print(f"        {o.detail}")
    counts = summarize(outcomes)
    print(f"{'-' * 70}")
    print(f"  PASS={counts['PASS']}  FAIL={counts['FAIL']}  SKIP={counts['SKIP']}")
    print(f"{'=' * 70}\n")
    return counts["FAIL"] == 0
