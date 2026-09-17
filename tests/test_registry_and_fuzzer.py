"""Registries populate and the config fuzzer expands list knobs × runs correctly."""

import subprocess
import sys

from fltest.core.config import TestConfig
from fltest.core.orchestrator import expand_run_specs


def test_registries_populated():
    import fltest.frameworks, fltest.attacks, fltest.defenses, fltest.metrics  # noqa: F401
    from fltest.core.registry import ATTACKS, DEFENSES, FRAMEWORKS, METRICS

    assert {"reference", "flwr"} <= set(FRAMEWORKS.names())
    assert {
        "label_flip", "sign_flip", "gaussian", "backdoor", "dlg",
        "membership_inference", "model_replacement",
    } <= set(ATTACKS.names())
    assert {"gradient_noise", "norm_clip", "krum", "trimmed_mean", "median", "fldetector"} <= set(DEFENSES.names())
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


def test_lazy_declarations_resolve():
    """Every lazily declared name must really be registered by the module it names.

    Guards against drift: renaming a plugin module or its registry name would otherwise
    only fail at run time, when a config asks for it.
    """
    from fltest.attacks import BUILTIN_ATTACKS
    from fltest.core.registry import ATTACKS, DEFENSES, FRAMEWORKS, METRICS
    from fltest.defenses import BUILTIN_DEFENSES
    from fltest.frameworks import BUILTIN_FRAMEWORKS
    from fltest.metrics import BUILTIN_METRICS

    for registry, declared in (
        (ATTACKS, BUILTIN_ATTACKS),
        (DEFENSES, BUILTIN_DEFENSES),
        (METRICS, BUILTIN_METRICS),
        (FRAMEWORKS, BUILTIN_FRAMEWORKS),
    ):
        for name in declared:
            assert registry.get(name) is not None, name


def test_listing_the_catalog_does_not_import_torch():
    """`fltest list` must stay fast, which means declaring names loads no heavy deps."""
    code = (
        "import sys;"
        "import fltest.frameworks, fltest.attacks, fltest.defenses, fltest.metrics;"
        "from fltest.core.registry import ATTACKS, DEFENSES, FRAMEWORKS, METRICS;"
        "[r.names() for r in (ATTACKS, DEFENSES, FRAMEWORKS, METRICS)];"
        "sys.exit(1 if 'torch' in sys.modules else 0)"
    )
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


def test_aggregation_names_the_rule_that_combines_updates():
    """FedAvg unless a robust-aggregation defense replaces it at before_aggregate."""
    from fltest.core.config import RunSpec

    plain = RunSpec(run_id="a", run_name="a", framework="reference")
    assert plain.aggregation() == "fedavg"

    robust = RunSpec(run_id="b", run_name="b", framework="reference",
                     defenses=[{"name": "median"}])
    assert robust.aggregation() == "median"
    assert robust.summary()["aggregation"] == "median"

    # A perturbation defense leaves the aggregation rule alone.
    noised = RunSpec(run_id="c", run_name="c", framework="reference",
                     defenses=[{"name": "gradient_noise"}])
    assert noised.aggregation() == "fedavg"
