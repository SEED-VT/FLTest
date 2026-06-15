"""Automated test orchestration: fuzz a config into runs, execute them, collect results.

* :func:`expand_run_specs` is the **config fuzzer** — it takes the cartesian product of
  every list-valued knob and crosses it with the ``runs`` block, yielding one flat
  :class:`RunSpec` per (param-set × framework) cell.
* :func:`prepare_data` builds the dataset/loaders for a spec (cached on disk).
* :class:`Orchestrator` runs each spec via its registered adapter and returns a
  :class:`RunMatrix` of results.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List

from fltest.core.config import (
    FUZZABLE_KNOBS,
    AttackSpec,
    DefenseSpec,
    RunSpec,
    TestConfig,
)
from fltest.core.wiring import build_hook_runner
from fltest.data.datasets import build_dataloaders, dataset_meta, get_cached_federated_dataset
from fltest.frameworks.base import RunResult, get_adapter


@dataclass
class RunMatrix:
    """All run results for a config, plus the config itself."""

    config: TestConfig
    specs: List[RunSpec]
    results: List[RunResult] = field(default_factory=list)
    total_duration: float = 0.0


def _as_list(v: Any) -> List[Any]:
    return v if isinstance(v, list) else [v]


def _short_hash(d: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:10]


def expand_run_specs(config: TestConfig) -> List[RunSpec]:
    """Expand a TestConfig into concrete RunSpecs (config fuzzing × runs)."""
    # 1. Cartesian product over list-valued fuzzable knobs.
    knob_values = {k: _as_list(getattr(config, k)) for k in FUZZABLE_KNOBS if hasattr(config, k)}
    knob_names = list(knob_values)
    combos = list(itertools.product(*[knob_values[k] for k in knob_names]))

    common = dict(
        seed=config.seed, device=config.device, deterministic=config.deterministic,
        total_cpus=config.total_cpus, total_gpus=config.total_gpus,
        dataset_partitions=config.dataset_partitions,
        server_batch_size=config.server_batch_size, max_test_data_size=config.max_test_data_size,
        loss_fn=config.loss_fn,
        model_cache_path=config.model_cache_path, dataset_cache_path=config.dataset_cache_path,
        fw_cache_path=config.fw_cache_path,
    )

    specs: List[RunSpec] = []
    for combo in combos:
        param_set = dict(zip(knob_names, combo))
        for run in config.runs:
            run = dict(run)
            framework = run.pop("framework", "reference")
            run_label = run.pop("name", framework)

            merged: Dict[str, Any] = {**common, **param_set, **run}
            channels, num_classes = dataset_meta(merged.get("dataset", "mnist"))
            merged["channels"], merged["num_classes"] = channels, num_classes

            # plugins: per-run override if present, else global config-level.
            attacks = run.get("attacks", [a.model_dump() for a in config.attacks])
            defenses = run.get("defenses", [d.model_dump() for d in config.defenses])
            merged["attacks"] = [AttackSpec(**a) if isinstance(a, dict) else a for a in attacks]
            merged["defenses"] = [DefenseSpec(**d) if isinstance(d, dict) else d for d in defenses]
            merged["metrics"] = run.get("metrics", list(config.metrics))

            identity = {k: merged[k] for k in sorted(merged) if k != "fw_cache_path"}
            identity["framework"] = framework
            run_id = _short_hash(identity)
            merged["run_id"] = run_id
            merged["run_name"] = f"{run_label}"
            merged["framework"] = framework
            specs.append(RunSpec(**merged))
    return specs


def prepare_data(spec: RunSpec) -> Dict[str, Any]:
    """Load + partition the dataset and build loaders for a spec (cached)."""
    dataset_dict = get_cached_federated_dataset(
        spec.dataset, spec.dataset_partitions, spec.dataset_cache_path,
        spec.data_distribution, **spec.partitioner_kwargs(),
    )
    loaders = build_dataloaders(
        dataset_dict, spec.num_clients, spec.client_batch_size,
        spec.server_batch_size, spec.max_test_data_size, spec.seed,
    )
    # Keep the raw HF shards available for the pitfall checker (label histograms).
    loaders["dataset_dict"] = dataset_dict
    return loaders


class Orchestrator:
    """Executes a list of RunSpecs and gathers their results."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[fltest] {msg}")

    def run_spec(self, spec: RunSpec) -> RunResult:
        adapter = get_adapter(spec.framework)
        data = prepare_data(spec)
        hook_runner = build_hook_runner(spec)
        start = time.time()
        try:
            result = adapter.run_simulation(spec, data, hook_runner)
        except Exception as exc:  # noqa: BLE001 - surface any backend failure as a result
            result = RunResult(
                run_id=spec.run_id, run_name=spec.run_name, framework=spec.framework,
                status="failed", error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
            )
        result.duration_seconds = time.time() - start
        result.params = spec.summary()  # annotate the result with its resolved parameters
        return result

    @staticmethod
    def _describe(spec: RunSpec) -> str:
        """One-line human descriptor of a run's distinguishing parameters."""
        dist = spec.data_distribution
        if spec.data_distribution == "dirichlet":
            dist += f"(α={spec.dirichlet_alpha})"
        elif spec.data_distribution == "pathological":
            dist += f"({spec.classes_per_partition}cls)"
        defense = ",".join(d.name for d in spec.defenses) or "none"
        return (f"data={spec.dataset}/{dist} clients={spec.num_clients} "
                f"rounds={spec.num_rounds} defense={defense}")

    def run(self, config: TestConfig) -> RunMatrix:
        specs = expand_run_specs(config)
        matrix = RunMatrix(config=config, specs=specs)
        start = time.time()
        for i, spec in enumerate(specs, 1):
            self._log(f"run {i}/{len(specs)}: {spec.run_name} [{spec.framework}] "
                      f"({self._describe(spec)}) run_id={spec.run_id}")
            result = self.run_spec(spec)
            status = result.status if result.status != "success" else (
                f"acc={result.final.get('accuracy', float('nan')):.4f}"
            )
            self._log(f"  -> {status} in {result.duration_seconds:.1f}s")
            matrix.results.append(result)
        matrix.total_duration = time.time() - start
        return matrix
