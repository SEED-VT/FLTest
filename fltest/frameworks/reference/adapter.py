"""Pure-PyTorch FedAvg adapter with the full FLTest hook lifecycle."""

from __future__ import annotations

import copy
from typing import Any, Dict, List

import numpy as np
import torch

from fltest.core import HookContext, HookRunner
from fltest.core.config import RunSpec
from fltest.core.registry import register_framework
from fltest.data.models import get_model, model_weight_sum, test, train
from fltest.data.utils import (
    aggregate_ndarrays,
    load_ndarrays_into,
    seed_everything,
    state_dict_to_ndarrays,
)
from fltest.frameworks.base import FrameworkAdapter, RunResult


@register_framework("reference")
class ReferenceAdapter(FrameworkAdapter):
    """Synchronous FedAvg over local DataLoaders. Deterministic on CPU."""

    name = "reference"

    def _new_model(self, spec: RunSpec):
        return get_model(
            spec.model_name,
            spec.model_cache_path,
            channels=spec.channels,
            num_classes=spec.num_classes,
            deterministic=spec.deterministic,
        ).to(spec.device)

    def run_simulation(self, spec: RunSpec, data: Dict[str, Any], hook_runner: HookRunner) -> RunResult:
        seed_everything(spec.seed)
        result = RunResult(run_id=spec.run_id, run_name=spec.run_name, framework=self.name)

        c2loader: Dict[int, Any] = dict(data["c2loader"])
        test_loader = data["test_loader"]
        history: Dict[int, Dict[str, Any]] = {}

        # --- before_simulation ---
        ctx = HookContext(
            cfg=spec, framework=self.name, run_name=spec.run_name,
            dist_dict=c2loader, test_data=test_loader, history=history,
        )
        hook_runner.run("before_simulation", ctx)

        # --- on_data_distribute (attacks may repartition / poison the mapping) ---
        ctx = HookContext(
            cfg=spec, framework=self.name, run_name=spec.run_name,
            dist_dict=c2loader, test_data=test_loader, history=history,
        )
        hook_runner.run("on_data_distribute", ctx)
        c2loader = ctx.dist_dict if ctx.dist_dict is not None else c2loader

        global_model = self._new_model(spec)
        global_params: List[np.ndarray] = state_dict_to_ndarrays(global_model.state_dict())

        for rnd in range(1, spec.num_rounds + 1):
            ctx_r = HookContext(
                cfg=spec, framework=self.name, run_name=spec.run_name, round=rnd,
                global_state=global_params, test_data=test_loader, history=history,
            )
            hook_runner.run("before_round", ctx_r)

            updates_and_weights: List[tuple] = []
            client_hook_metrics: Dict[str, Any] = {}
            for cid in range(spec.num_clients):
                loader = c2loader[cid]

                # before_client_train: attacks may swap the loader; DLG reconstructs here.
                ctx_pre = HookContext(
                    cfg=spec, framework=self.name, run_name=spec.run_name, round=rnd,
                    client_id=cid, client_data=loader, global_state=global_params,
                    test_data=test_loader, history=history,
                )
                hook_runner.run("before_client_train", ctx_pre)
                loader = ctx_pre.client_data if ctx_pre.client_data is not None else loader
                client_hook_metrics.update(ctx_pre.metrics)

                # local training from the current global weights
                local = self._new_model(spec)
                load_ndarrays_into(local, global_params)
                train(
                    local, loader, epochs=spec.client_epochs, device=spec.device,
                    loss_fn=spec.loss_fn, optimizer_name=spec.optimizer, lr=spec.client_lr,
                )
                update = state_dict_to_ndarrays(local.state_dict())
                n_samples = sum(b["label"].size(0) for b in loader)

                # after_client_train: defenses (clip+noise), attacks may tamper the update.
                ctx_post = HookContext(
                    cfg=spec, framework=self.name, run_name=spec.run_name, round=rnd,
                    client_id=cid, client_update=update, num_samples=n_samples,
                    global_state=global_params, model=local, test_data=test_loader, history=history,
                )
                hook_runner.run("after_client_train", ctx_post)
                update = ctx_post.client_update if ctx_post.client_update is not None else update
                client_hook_metrics.update(ctx_post.metrics)
                updates_and_weights.append((update, n_samples))

            # --- aggregation ---
            ctx_agg = HookContext(
                cfg=spec, framework=self.name, run_name=spec.run_name, round=rnd,
                updates_and_weights=updates_and_weights, global_state=global_params, history=history,
            )
            hook_runner.run("before_aggregate", ctx_agg)
            uw = ctx_agg.updates_and_weights if ctx_agg.updates_and_weights is not None else updates_and_weights
            global_params = aggregate_ndarrays(uw)
            ctx_agg.new_global_state = global_params
            hook_runner.run("on_aggregate", ctx_agg)
            hook_runner.run("after_aggregate", ctx_agg)
            client_hook_metrics.update(ctx_agg.metrics)  # e.g. server-side DLG, robust-agg stats

            # --- evaluate global model + after_round (metric listeners record here) ---
            load_ndarrays_into(global_model, global_params)
            loss, acc = test(global_model, test_loader, device=spec.device, loss_fn=spec.loss_fn)
            ctx_round = HookContext(
                cfg=spec, framework=self.name, run_name=spec.run_name, round=rnd,
                global_state=global_params, model=global_model, test_data=test_loader,
                history=history,
            )
            ctx_round.record(loss=loss, accuracy=acc, gm_weight_sum=model_weight_sum(global_model))
            # surface client-side hook metrics (e.g. DLG reconstruction, backdoor ASR)
            ctx_round.record(**client_hook_metrics)
            hook_runner.run("after_round", ctx_round)
            history[rnd] = dict(ctx_round.metrics)

        # --- after_simulation ---
        final_round = max(history) if history else 0
        final_metrics = dict(history.get(final_round, {}))
        ctx_final = HookContext(
            cfg=spec, framework=self.name, run_name=spec.run_name,
            global_state=global_params, model=global_model, test_data=test_loader,
            metrics=final_metrics, history=history, final_accuracy=final_metrics.get("accuracy"),
        )
        hook_runner.run("after_simulation", ctx_final)

        result.history = history
        result.final = dict(ctx_final.metrics)
        result.extras = dict(ctx_final.extras)
        return result
