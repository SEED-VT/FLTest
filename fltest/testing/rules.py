"""Relation rules used by differential and metamorphic testing.

Each rule takes evidence and returns ``(passed: bool, detail: str)``. They are deliberately
tiny and pure so they are trivial to unit-test and reuse.
"""

from __future__ import annotations

from typing import List, Tuple

Pair = Tuple[float, float]  # (parameter_value, metric_value)


def parity(values: List[float], tolerance: float) -> Tuple[bool, str]:
    """All values must lie within ``tolerance`` of each other (differential parity)."""
    if len(values) < 2:
        return True, "fewer than 2 values; trivially parity"
    spread = max(values) - min(values)
    ok = spread <= tolerance
    return ok, f"max|Δ|={spread:.4f} (tol={tolerance})"


def non_decreasing(pairs: List[Pair], tolerance: float) -> Tuple[bool, str]:
    """Metric must not *decrease* as the parameter increases (within tolerance)."""
    pairs = sorted(pairs, key=lambda p: p[0])
    violations = [
        (a[0], b[0], b[1] - a[1])
        for a, b in zip(pairs, pairs[1:])
        if b[1] < a[1] - tolerance
    ]
    if violations:
        return False, f"{len(violations)} non-decreasing violation(s): {violations}"
    return True, f"non-decreasing over {[p[0] for p in pairs]}"


def non_increasing(pairs: List[Pair], tolerance: float) -> Tuple[bool, str]:
    """Metric must not *increase* as the parameter increases (within tolerance)."""
    pairs = sorted(pairs, key=lambda p: p[0])
    violations = [
        (a[0], b[0], b[1] - a[1])
        for a, b in zip(pairs, pairs[1:])
        if b[1] > a[1] + tolerance
    ]
    if violations:
        return False, f"{len(violations)} non-increasing violation(s): {violations}"
    return True, f"non-increasing over {[p[0] for p in pairs]}"


def no_drop(reference: float, candidate: float, tolerance: float) -> Tuple[bool, str]:
    """``candidate`` must not fall more than ``tolerance`` below ``reference``."""
    drop = reference - candidate
    ok = drop <= tolerance
    return ok, f"reference={reference:.4f} candidate={candidate:.4f} drop={drop:.4f} (tol={tolerance})"
