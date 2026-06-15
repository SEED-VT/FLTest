"""Configuration schema for FLTest (``test_conf.yaml``).

Two layers:

* :class:`TestConfig` — the raw top-level document. Any FL knob may be a scalar *or a
  list*; a list marks that knob for **config fuzzing** (the orchestrator expands the
  cartesian product of all list-valued knobs). The ``runs`` block lists the frameworks
  (and optional per-run overrides) to execute — this is what enables cross-framework
  differential testing on one config.
* :class:`RunSpec` — a single fully-resolved, flat run that a :class:`FrameworkAdapter`
  consumes. The orchestrator produces one ``RunSpec`` per (fuzzed param-set × run) cell.

Derived fields (``channels``, ``num_classes``) come from the dataset metadata so users
never specify them by hand.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, Field

from fltest.data.datasets import dataset_meta

ScalarOrList = Union[Any, List[Any]]


class AttackSpec(BaseModel):
    """A built-in attack to compose onto a run (by registry name)."""

    name: str
    params: Dict[str, Any] = Field(default_factory=dict)
    target_clients: Optional[List[int]] = None  # None => attack decides (often all)


class DefenseSpec(BaseModel):
    """A built-in defense/PPFL technique to compose onto a run (by registry name)."""

    name: str
    params: Dict[str, Any] = Field(default_factory=dict)


class DifferentialSpec(BaseModel):
    """Differential-testing settings (slide 21)."""

    enabled: bool = True
    # cross_framework: compare final metric across runs that differ only by framework.
    # determinism: re-run each spec and require identical results.
    mode: str = "cross_framework"
    metric: str = "accuracy"
    tolerance: float = 0.05


class MetamorphicRelation(BaseModel):
    """One metamorphic relation to check (slide 22).

    ``relation`` selects the transform+rule pair, e.g.:
      * ``clients_scale``    — N -> 2N clients (IID) must not drop accuracy beyond tolerance.
      * ``rounds_monotonic`` — more rounds must not decrease accuracy beyond tolerance.
      * ``attack_strength``  — stronger attack must not *increase* accuracy (non_increasing).
      * ``dp_noise``         — more DP noise must not increase accuracy (utility non_increasing).
    """

    relation: str
    parameter: Optional[str] = None
    metric: str = "accuracy"
    values: Optional[List[Any]] = None
    tolerance: float = 0.05


class TestingSpec(BaseModel):
    differential: Optional[DifferentialSpec] = None
    metamorphic: List[MetamorphicRelation] = Field(default_factory=list)


# Knobs that participate in config fuzzing when given as a list.
FUZZABLE_KNOBS = (
    "dataset",
    "data_distribution",
    "model_name",
    "num_clients",
    "num_rounds",
    "client_epochs",
    "client_lr",
    "client_batch_size",
    "dirichlet_alpha",
    "classes_per_partition",
    "optimizer",
)


class RunSpec(BaseModel):
    """A fully-resolved single run consumed by a FrameworkAdapter."""

    run_id: str
    run_name: str
    framework: str

    # reproducibility / hardware
    seed: int = 786
    device: str = "cpu"
    deterministic: bool = True
    total_cpus: int = 4
    total_gpus: int = 0

    # data
    dataset: str = "mnist"
    data_distribution: str = "iid"
    dirichlet_alpha: float = 0.5
    classes_per_partition: int = 2
    dataset_partitions: int = 100  # how finely to partition before selecting num_clients shards

    # model
    model_name: str = "LeNet"
    channels: int = 1
    num_classes: int = 10

    # FL params
    num_clients: int = 10
    num_rounds: int = 10
    client_epochs: int = 1
    client_lr: float = 0.01
    client_batch_size: int = 32
    server_batch_size: int = 256
    max_test_data_size: int = 2048
    optimizer: str = "SGD"
    loss_fn: str = "CrossEntropyLoss"

    # composed plugins
    attacks: List[AttackSpec] = Field(default_factory=list)
    defenses: List[DefenseSpec] = Field(default_factory=list)
    metrics: List[str] = Field(default_factory=lambda: ["accuracy", "loss"])

    # cache paths
    model_cache_path: str = "data/models_cache"
    dataset_cache_path: str = "data/dataset_cache"
    fw_cache_path: str = "data/caches"

    extras: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "allow"}

    def partitioner_kwargs(self) -> Dict[str, Any]:
        if self.data_distribution == "dirichlet":
            return {"alpha": self.dirichlet_alpha}
        if self.data_distribution == "pathological":
            return {"classes_per_partition": self.classes_per_partition}
        return {}

    def summary(self) -> Dict[str, Any]:
        """Compact, fully-resolved parameter annotation for reports and logs.

        Distribution-specific knobs are only included when they apply, so e.g.
        ``dirichlet_alpha`` is None for IID runs (where it has no effect).
        """
        return {
            "framework": self.framework,
            "dataset": self.dataset,
            "data_distribution": self.data_distribution,
            "dirichlet_alpha": self.dirichlet_alpha if self.data_distribution == "dirichlet" else None,
            "classes_per_partition": (
                self.classes_per_partition if self.data_distribution == "pathological" else None
            ),
            "model_name": self.model_name,
            "num_clients": self.num_clients,
            "num_rounds": self.num_rounds,
            "client_epochs": self.client_epochs,
            "client_lr": self.client_lr,
            "client_batch_size": self.client_batch_size,
            "optimizer": self.optimizer,
            "seed": self.seed,
            "attacks": [
                {"name": a.name, "params": a.params, "target_clients": a.target_clients}
                for a in self.attacks
            ],
            "defenses": [{"name": d.name, "params": d.params} for d in self.defenses],
            "metrics": list(self.metrics),
        }


class TestConfig(BaseModel):
    """Raw top-level ``test_conf.yaml``. Scalar fields may be lists (fuzzed)."""

    __test__ = False  # not a pytest test class despite the "Test" prefix

    name: str = "fltest_run"
    seed: int = 786
    device: str = "cpu"
    deterministic: bool = True
    total_cpus: int = 4
    total_gpus: int = 0

    dataset: ScalarOrList = "mnist"
    data_distribution: ScalarOrList = "iid"
    dirichlet_alpha: ScalarOrList = 0.5
    classes_per_partition: ScalarOrList = 2
    dataset_partitions: int = 100

    model_name: ScalarOrList = "LeNet"
    num_clients: ScalarOrList = 10
    num_rounds: ScalarOrList = 10
    client_epochs: ScalarOrList = 1
    client_lr: ScalarOrList = 0.01
    client_batch_size: ScalarOrList = 32
    server_batch_size: int = 256
    max_test_data_size: int = 2048
    optimizer: ScalarOrList = "SGD"
    loss_fn: str = "CrossEntropyLoss"

    # One run per framework (with optional per-run overrides). Defaults to a single
    # reference run if omitted.
    runs: List[Dict[str, Any]] = Field(default_factory=lambda: [{"framework": "reference"}])

    attacks: List[AttackSpec] = Field(default_factory=list)
    defenses: List[DefenseSpec] = Field(default_factory=list)
    metrics: List[str] = Field(default_factory=lambda: ["accuracy", "loss"])
    testing: TestingSpec = Field(default_factory=TestingSpec)

    model_cache_path: str = "data/models_cache"
    dataset_cache_path: str = "data/dataset_cache"
    fw_cache_path: str = "data/caches"

    model_config = {"extra": "allow"}

    def derived_channels_classes(self, dataset_name: str):
        return dataset_meta(dataset_name)


def load_config(path: str | Path) -> TestConfig:
    """Load and validate a ``test_conf.yaml`` into a :class:`TestConfig`."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open() as fh:
        data = yaml.safe_load(fh) or {}
    return TestConfig(**data)
