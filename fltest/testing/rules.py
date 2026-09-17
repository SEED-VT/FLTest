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


def exactly_equal(pairs: List[Pair], tolerance: float) -> Tuple[bool, str]:
    """Metric must be *identical* across every parameter value.

    The other rules here are inequalities with slack, which is all an accuracy-based oracle
    can support. This one is an equality: it applies where the transform provably must not
    move the result at all — changing a secure-aggregation mask seed, for instance, since
    the masks are supposed to cancel whatever they are. Pair it with ``tolerance: 0.0`` and
    a metric that fingerprints the model (``gm_weight_sum``) rather than one that projects
    it through a test set (``accuracy``), and any residue that fails to cancel shows up.
    """
    values = [v for _, v in pairs]
    if len(values) < 2:
        return True, "fewer than 2 values; trivially equal"
    spread = max(values) - min(values)
    ok = spread <= tolerance
    detail = f"spread={spread:.6g} over {[p[0] for p in pairs]} (tol={tolerance})"
    return ok, detail if ok else f"values differ: {detail}"
