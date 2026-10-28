"""Flower backend: drives ``flwr.simulation.run_simulation`` through the hook lifecycle.

Driver-side hooks (``before_simulation``, ``on_data_distribute``, server-side round hooks,
``after_simulation``) run in this process via ``driver_runner``. Client-side hooks run in
Ray workers (rebuilt from the spec in :mod:`fltest.frameworks.flower.client`).
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from flwr.simulation import run_simulation

from fltest.core import HookContext, HookRunner
from fltest.core.config import RunSpec
from fltest.core.registry import register_framework
from fltest.core.wiring import build_hook_runner
from fltest.data.utils import seed_everything, state_dict_to_ndarrays
from fltest.data.models import get_model
from fltest.frameworks.base import FrameworkAdapter, RunResult
from fltest.frameworks.flower.client import get_client_app
from fltest.frameworks.flower.server import get_server_app


RAY_ADDRESS_ENV = "FLTEST_RAY_ADDRESS"


def ray_cluster_address() -> Optional[str]:
    """Address of an existing Ray cluster to run Flower clients on, or None for a local one.

    Set ``FLTEST_RAY_ADDRESS`` (for example ``10.0.0.1:6379``) to spread the simulated clients
    over every node of a running Ray cluster instead of the machine running ``fltest``.
    """
    return os.environ.get(RAY_ADDRESS_ENV) or None


def ray_backend_config(total_cpus: int, total_gpus: int, num_clients: int) -> Dict[str, Any]:
    """Flower's Ray backend settings: a private local Ray sized by the spec, or, when
    ``FLTEST_RAY_ADDRESS`` is set, a connection to that cluster. Ray refuses resource
    arguments when it joins an existing cluster, since the cluster already declares them."""
    per_client_gpu = (total_gpus / num_clients) if total_gpus else 0.0
    address = ray_cluster_address()
    init_args = {"address": address} if address else {"num_cpus": total_cpus, "num_gpus": total_gpus}
    return {"client_resources": {"num_cpus": 1, "num_gpus": per_client_gpu}, "init_args": init_args}


@register_framework("flwr")
@register_framework("flower")
class FlowerAdapter(FrameworkAdapter):
    name = "flwr"

    def run_simulation(self, spec: RunSpec, data: Dict[str, Any], hook_runner: HookRunner) -> RunResult:
        seed_everything(spec.seed)
        result = RunResult(run_id=spec.run_id, run_name=spec.run_name, framework=self.name)

        c2loader = dict(data["c2loader"])
        test_loader = data["test_loader"]
        history: Dict[int, Dict[str, Any]] = {}

        # Driver-side hook runner (server-side + lifecycle hooks run in this process).
        driver_runner = build_hook_runner(spec)

        init_model = get_model(
            spec.model_name, spec.model_cache_path, channels=spec.channels,
            num_classes=spec.num_classes, deterministic=spec.deterministic,
        )
        init_params = state_dict_to_ndarrays(init_model.state_dict())

        driver_runner.run("before_simulation", HookContext(
            cfg=spec, framework=self.name, run_name=spec.run_name,
            dist_dict=c2loader, test_data=test_loader, history=history, global_state=init_params,
        ))
        ctx_data = HookContext(
            cfg=spec, framework=self.name, run_name=spec.run_name,
            dist_dict=c2loader, test_data=test_loader, history=history,
        )
        driver_runner.run("on_data_distribute", ctx_data)
        c2loader = ctx_data.dist_dict if ctx_data.dist_dict is not None else c2loader

        backend_config = ray_backend_config(spec.total_cpus, spec.total_gpus, spec.num_clients)

        state_holder: Dict[str, Any] = {}
        server_app = get_server_app(spec, driver_runner, history, test_loader, state_holder)
        client_app = get_client_app(spec, c2loader)

        run_simulation(
            server_app=server_app,
            client_app=client_app,
            num_supernodes=spec.num_clients,
            backend_config=backend_config,
        )

        final_round = max(history) if history else 0
        final_metrics = dict(history.get(final_round, {}))
        ctx_final = HookContext(
            cfg=spec, framework=self.name, run_name=spec.run_name,
            test_data=test_loader, metrics=final_metrics, history=history,
            final_accuracy=final_metrics.get("accuracy"),
        )
        # per_client listener needs the loaders it cached at on_data_distribute + final params.
        ctx_final.global_state = state_holder.get("final_params")
        driver_runner.run("after_simulation", ctx_final)

        result.history = history
        result.final = dict(ctx_final.metrics)
        result.extras = dict(ctx_final.extras)
        return result
