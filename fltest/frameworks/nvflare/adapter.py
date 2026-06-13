"""NVIDIA FLARE FedAvg backend.

NVFlare runs each client in its own simulator process via a ScriptRunner, so FLTest
client-side hooks (attacks/defenses at ``before/after_client_train``) are **not** supported
here — this backend is for *cross-framework differential* parity of the vanilla FedAvg path.
Driver-side hooks (``before_simulation``, per-round ``after_round``, ``after_simulation``)
run in this process, and per-round/final accuracy is evaluated against the shared test
loader so the metric is comparable to the reference/Flower backends.

The custom controller snapshots the aggregated global model to the run cache each round;
after the simulation we replay those snapshots to emit ``after_round`` with real metrics.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from diskcache import Index
from nvflare.app_common.workflows.fedavg import FedAvg

from fltest.core import HookContext, HookRunner
from fltest.core.config import RunSpec
from fltest.core.registry import register_framework
from fltest.data.models import get_model, model_weight_sum, test
from fltest.data.utils import seed_everything, state_dict_to_ndarrays
from fltest.frameworks.base import FrameworkAdapter, RunResult

_CLIENT_SCRIPT = str(Path(__file__).resolve().parent / "client_script.py")


class SnapshotFedAvg(FedAvg):
    """FedAvg controller that snapshots the aggregated global model to the run cache
    each round, so the adapter can replay rounds and emit ``after_round`` with metrics.

    Must be module-level: NVFlare serializes the controller by import path and rebuilds
    it in the server process.
    """

    def __init__(self, fltest_cache_path: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fltest_cache_path = fltest_cache_path

    def run(self):
        cache = Index(self.fltest_cache_path)
        model = self.load_model()
        model.start_round = self.start_round
        model.total_rounds = self.num_rounds
        for self.current_round in range(self.start_round, self.start_round + self.num_rounds):
            model.current_round = self.current_round
            clients = self.sample_clients(self.num_clients)
            results = self.send_model_and_wait(targets=clients, data=model)
            agg = self.aggregate(results, aggregate_fn=self.aggregate_fn)
            model = self.update_model(model, agg)
            cache[f"round_{self.current_round}"] = {"gm": model.params}
            self.save_model(model)


@register_framework("nvflare")
@register_framework("flare")
class NVFlareAdapter(FrameworkAdapter):
    name = "nvflare"

    def run_simulation(self, spec: RunSpec, data: Dict[str, Any], hook_runner: HookRunner) -> RunResult:
        from nvflare.app_opt.pt.job_config.base_fed_job import BaseFedJob
        from nvflare.job_config.script_runner import ScriptRunner

        seed_everything(spec.seed)
        result = RunResult(run_id=spec.run_id, run_name=spec.run_name, framework=self.name)

        dataset_dict = data["dataset_dict"]
        test_loader = data["test_loader"]
        history: Dict[int, Dict[str, Any]] = {}

        # NVFlare requires absolute paths.
        cache_path = os.path.abspath(os.path.join(spec.fw_cache_path, "nvflare_cache"))
        workspace = os.path.abspath(os.path.join(spec.fw_cache_path, "nvflare_workspace"))
        model_cache_abs = os.path.abspath(spec.model_cache_path)
        shutil.rmtree(workspace, ignore_errors=True)
        os.makedirs(workspace, exist_ok=True)

        spec_abs = spec.model_copy(update={"model_cache_path": model_cache_abs})
        cache = Index(cache_path)
        cache["nvflare_spec"] = spec_abs
        cache["nvflare_dataset"] = dataset_dict

        ctx0 = HookContext(cfg=spec, framework=self.name, run_name=spec.run_name,
                           dist_dict=dataset_dict["c2data"], test_data=test_loader, history=history)
        hook_runner.run("before_simulation", ctx0)
        hook_runner.run("on_data_distribute", ctx0)

        initial_model = get_model(spec.model_name, model_cache_abs, channels=spec.channels,
                                  num_classes=spec.num_classes, deterministic=spec.deterministic)
        controller = SnapshotFedAvg(
            fltest_cache_path=cache_path, num_clients=spec.num_clients, num_rounds=spec.num_rounds)
        job = BaseFedJob(name=f"fltest_{spec.run_id}", initial_model=initial_model)
        job.to(controller, "server")
        for cid in range(spec.num_clients):
            runner = ScriptRunner(script=_CLIENT_SCRIPT,
                                  script_args=f"--client_id {cid} --cache_path {cache_path}")
            job.to(runner, f"site-{cid}")
        job.simulator_run(workspace)

        # Replay per-round snapshots to emit after_round with real metrics.
        eval_model = get_model(spec.model_name, model_cache_abs, channels=spec.channels,
                               num_classes=spec.num_classes, deterministic=spec.deterministic).to(spec.device)
        for rnd in range(spec.num_rounds):
            snap = cache.get(f"round_{rnd}")
            if snap is None:
                continue
            torch_sd = {k: torch.as_tensor(v) if isinstance(v, np.ndarray) else v
                        for k, v in snap["gm"].items()}
            eval_model.load_state_dict(torch_sd)
            loss, acc = test(eval_model, test_loader, device=spec.device, loss_fn=spec.loss_fn)
            ctx = HookContext(cfg=spec, framework=self.name, run_name=spec.run_name, round=rnd + 1,
                              global_state=state_dict_to_ndarrays(eval_model.state_dict()),
                              model=eval_model, test_data=test_loader, history=history)
            ctx.record(loss=loss, accuracy=acc, gm_weight_sum=model_weight_sum(eval_model))
            hook_runner.run("after_round", ctx)
            history[rnd + 1] = dict(ctx.metrics)

        final_round = max(history) if history else 0
        final_metrics = dict(history.get(final_round, {}))
        ctx_final = HookContext(cfg=spec, framework=self.name, run_name=spec.run_name,
                                global_state=state_dict_to_ndarrays(eval_model.state_dict()),
                                model=eval_model, test_data=test_loader, metrics=final_metrics,
                                history=history, final_accuracy=final_metrics.get("accuracy"))
        hook_runner.run("after_simulation", ctx_final)

        result.history = history
        result.final = dict(ctx_final.metrics)
        if not result.final:
            result.status = "failed"
            result.error = "NVFlare produced no round snapshots (see simulator logs in workspace)."
        return result
