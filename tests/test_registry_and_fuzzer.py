"""Registries populate and the config fuzzer expands list knobs × runs correctly."""

from fltest.core.config import TestConfig
from fltest.core.orchestrator import expand_run_specs


def test_registries_populated():
    import fltest.frameworks, fltest.attacks, fltest.defenses, fltest.metrics  # noqa: F401
    from fltest.core.registry import ATTACKS, DEFENSES, FRAMEWORKS, METRICS

    assert {"reference", "flwr"} <= set(FRAMEWORKS.names())
    assert {"label_flip", "sign_flip", "gaussian", "backdoor", "dlg"} <= set(ATTACKS.names())
    assert {"gradient_noise", "norm_clip", "krum", "trimmed_mean", "median"} <= set(DEFENSES.names())
    assert {"accuracy", "loss", "per_client"} <= set(METRICS.names())


def test_fuzzer_expands_grid_times_runs():
    cfg = TestConfig(
        name="fuzz",
        num_clients=[2, 4],            # 2 values
        data_distribution=["iid", "dirichlet"],  # 2 values
        runs=[{"framework": "reference"}, {"framework": "flwr"}],  # 2 runs
    )
    specs = expand_run_specs(cfg)
    assert len(specs) == 2 * 2 * 2  # grid (4) × runs (2)
    assert {s.framework for s in specs} == {"reference", "flwr"}
    assert {s.num_clients for s in specs} == {2, 4}


def test_derived_channels_classes():
    cfg = TestConfig(name="d", dataset="cifar10", runs=[{"framework": "reference"}])
    spec = expand_run_specs(cfg)[0]
    assert spec.channels == 3 and spec.num_classes == 10
