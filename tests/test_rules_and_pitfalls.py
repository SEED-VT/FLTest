"""Unit tests for testing rules and the pitfall checker."""

from fltest.core.config import TestConfig
from fltest.pitfalls import check_config, recommend
from fltest.testing.rules import no_drop, non_decreasing, non_increasing, parity


def test_parity_rule():
    ok, _ = parity([0.90, 0.92, 0.91], tolerance=0.05)
    assert ok
    ok, _ = parity([0.90, 0.40], tolerance=0.05)
    assert not ok


def test_monotonic_rules_with_tolerance():
    assert non_decreasing([(1, 0.5), (2, 0.6), (3, 0.58)], tolerance=0.05)[0]
    assert not non_decreasing([(1, 0.8), (2, 0.5)], tolerance=0.05)[0]
    assert non_increasing([(0.0, 0.9), (0.1, 0.7), (0.2, 0.71)], tolerance=0.05)[0]
    assert not non_increasing([(0.0, 0.5), (0.1, 0.9)], tolerance=0.05)[0]


def test_no_drop_rule():
    assert no_drop(reference=0.90, candidate=0.88, tolerance=0.05)[0]
    assert not no_drop(reference=0.90, candidate=0.70, tolerance=0.05)[0]


def test_pitfall_checker_flags_weak_setup():
    cfg = TestConfig(name="weak", dataset="mnist", data_distribution="iid",
                     metrics=["accuracy", "loss"], runs=[{"framework": "reference"}])
    findings = check_config(cfg)
    ids = {f.pitfall for f in findings}
    assert "P1_threat_models" in ids       # no attacks
    assert "P3_iid_only" in ids            # iid only
    assert "P3_no_personalized" in ids     # no per_client metric
    recs = recommend(findings)
    assert recs and recs[0]["severity"] == "high"


def test_pitfall_checker_quiet_on_strong_setup():
    cfg = TestConfig(
        name="strong",
        dataset=["mnist", "cifar10"],
        data_distribution=["iid", "dirichlet"],
        metrics=["accuracy", "loss", "per_client"],
        attacks=[{"name": "backdoor"}, {"name": "dlg"}],
        runs=[{"framework": "reference"}],
    )
    ids = {f.pitfall for f in check_config(cfg)}
    assert "P1_threat_models" not in ids
    assert "P3_iid_only" not in ids
    assert "P5_subtle_leakage" not in ids
