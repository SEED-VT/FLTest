"""Flower backend: drives ``flwr.simulation.run_simulation`` through the hook lifecycle.

Driver-side hooks (``before_simulation``, ``on_data_distribute``, server-side round hooks,
``after_simulation``) run in this process via ``driver_runner``. Client-side hooks run in
Ray workers (rebuilt from the spec in :mod:`fltest.frameworks.flower.client`).
"""

from __future__ import annotations

from typing import Any, Dict

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

        num_gpus = spec.total_gpus
        per_client_gpu = (num_gpus / spec.num_clients) if num_gpus else 0.0
        backend_config = {
            "client_resources": {"num_cpus": 1, "num_gpus": per_client_gpu},
            "init_args": {"num_cpus": spec.total_cpus, "num_gpus": num_gpus},
        }

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
