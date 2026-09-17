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


def test_exactly_equal_rule():
    from fltest.testing.rules import exactly_equal

    assert exactly_equal([(1, 0.5), (2, 0.5), (3, 0.5)], tolerance=0.0)[0]
    # The slack the monotonic rules allow is exactly what this rule must not allow.
    assert not exactly_equal([(1, 0.5), (2, 0.5001)], tolerance=0.0)[0]
    assert exactly_equal([(1, 0.5)], tolerance=0.0)[0]  # single value, nothing to compare


def _secagg_cfg(**overrides):
    """A config that clears every pitfall except the secure-aggregation ones under test."""
    base = dict(
        name="secagg", dataset="femnist", data_distribution=["iid", "dirichlet"],
        attacks=[{"name": "dlg"}], metrics=["accuracy", "loss", "per_client"],
        runs=[{"framework": "reference"}],
        testing={"metamorphic": [{"relation": "secagg_lossless", "parameter": "defense.seed",
                                  "values": [1, 2], "metric": "gm_weight_sum", "tolerance": 0.0}]},
    )
    base.update(overrides)
    return TestConfig(**base)


def test_pitfall_checker_flags_disabled_secagg_masks():
    cfg = _secagg_cfg(defenses=[{"name": "secure_aggregation", "params": {"mask_scale": 0}}])
    assert "P4_misconfig_secagg" in {f.pitfall for f in check_config(cfg)}


def test_pitfall_checker_flags_mpc_misconfiguration():
    cfg = _secagg_cfg(defenses=[{"name": "mpc_aggregation", "params": {
        "quant_bits": 4, "modulus": 16, "dropout_rate": 0.2}}])
    titles = {f.title for f in check_config(cfg) if f.pitfall == "P4_misconfig_secagg"}
    assert titles == {
        "MPC dropouts without recovery",
        "Fixed-point precision likely too low",
        "MPC ring too small for its precision",
    }


def test_pitfall_checker_flags_secagg_without_equality_oracle():
    cfg = _secagg_cfg(defenses=[{"name": "secure_aggregation"}], testing={"metamorphic": []})
    assert "P4_untested_secagg" in {f.pitfall for f in check_config(cfg)}
    # ... and stays quiet once the relation is configured.
    assert "P4_untested_secagg" not in {
        f.pitfall for f in check_config(_secagg_cfg(defenses=[{"name": "secure_aggregation"}]))
    }


def test_pitfall_checker_flags_masking_combined_with_robust_aggregation():
    """A masked server cannot compare updates client-by-client; the simulation lets it."""
    cfg = _secagg_cfg(defenses=[{"name": "secure_aggregation"}, {"name": "median"}])
    assert "P4_secagg_vs_robust" in {f.pitfall for f in check_config(cfg)}
