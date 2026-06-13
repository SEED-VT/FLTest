"""Flower ServerApp + hooked FedAvg strategy.

``HookedFedAvg`` aggregates with FLTest's own ``aggregate_ndarrays`` (the same weighted
mean the reference backend uses) so the two backends differ only in execution machinery,
not aggregation math — tightening cross-framework differential parity. It emits the
server-side lifecycle hooks (``before_round`` → ``after_round``); robust-aggregation
defenses that replace ``ctx.updates_and_weights`` at ``before_aggregate`` therefore take
effect here too.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from flwr.common import Code, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

from fltest.core import HookContext, HookRunner
from fltest.core.config import RunSpec
from fltest.data.models import get_model, model_weight_sum, test
from fltest.data.utils import aggregate_ndarrays
from fltest.frameworks.flower.utils import get_parameters, set_parameters


class HookedFedAvg(FedAvg):
    def __init__(self, hook_runner: HookRunner, spec: RunSpec, history: Dict, **kwargs):
        super().__init__(**kwargs)
        self._hooks = hook_runner
        self._spec = spec
        self._history = history
        self._last_fit_metrics: Dict[str, float] = {}

    def aggregate_fit(self, server_round, results, failures):
        successful = [(p, r) for p, r in results if r.status.code == Code.OK]
        if not successful:
            return None, {}

        updates_and_weights = [
            (parameters_to_ndarrays(r.parameters), r.num_examples) for _, r in successful
        ]

        self._hooks.run("before_round", HookContext(
            cfg=self._spec, framework="flwr", run_name=self._spec.run_name, round=server_round,
        ))

        ctx = HookContext(
            cfg=self._spec, framework="flwr", run_name=self._spec.run_name, round=server_round,
            updates_and_weights=updates_and_weights,
        )
        self._hooks.run("before_aggregate", ctx)
        uw = ctx.updates_and_weights if ctx.updates_and_weights is not None else updates_and_weights

        aggregated = aggregate_ndarrays(uw)
        ctx.new_global_state = aggregated
        self._hooks.run("on_aggregate", ctx)
        self._hooks.run("after_aggregate", ctx)

        # Surface client-side hook metrics (e.g. DLG reconstruction) by averaging numerics.
        acc: Dict[str, List[float]] = defaultdict(list)
        for _, r in successful:
            for k, v in r.metrics.items():
                if k != "cid" and isinstance(v, (int, float)):
                    acc[k].append(float(v))
        self._last_fit_metrics = {k: sum(v) / len(v) for k, v in acc.items()}

        return ndarrays_to_parameters(aggregated), dict(self._last_fit_metrics)


def get_server_app(spec: RunSpec, hook_runner: HookRunner, history: Dict, test_loader, state_holder: Dict) -> ServerApp:
    init_model = get_model(
        spec.model_name, spec.model_cache_path, channels=spec.channels,
        num_classes=spec.num_classes, deterministic=spec.deterministic,
    )
    initial_parameters = ndarrays_to_parameters(get_parameters(init_model))

    def evaluate_fn(server_round, parameters, config):
        net = get_model(
            spec.model_name, spec.model_cache_path, channels=spec.channels,
            num_classes=spec.num_classes, deterministic=spec.deterministic,
        ).to(spec.device)
        set_parameters(net, parameters)
        state_holder["final_params"] = list(parameters)  # for driver-side after_simulation hooks
        loss, acc = test(net, test_loader, device=spec.device, loss_fn=spec.loss_fn)

        ctx = HookContext(
            cfg=spec, framework="flwr", run_name=spec.run_name, round=server_round,
            global_state=parameters, model=net, test_data=test_loader, history=history,
        )
        ctx.record(loss=loss, accuracy=acc, gm_weight_sum=model_weight_sum(net))
        # merge client-side fit metrics captured during aggregation of this round
        strat_metrics = getattr(evaluate_fn, "_strategy", None)
        if strat_metrics is not None:
            ctx.metrics.update(strat_metrics._last_fit_metrics)
        hook_runner.run("after_round", ctx)
        if server_round > 0:
            history[server_round] = dict(ctx.metrics)
        return loss, {"accuracy": acc}

    def server_fn(context):
        strategy = HookedFedAvg(
            hook_runner=hook_runner, spec=spec, history=history,
            fraction_fit=1.0, fraction_evaluate=0.0,
            min_fit_clients=spec.num_clients, min_evaluate_clients=0,
            min_available_clients=spec.num_clients,
            evaluate_fn=evaluate_fn, initial_parameters=initial_parameters,
        )
        evaluate_fn._strategy = strategy  # let evaluate read the round's fit metrics
        return ServerAppComponents(strategy=strategy, config=ServerConfig(num_rounds=spec.num_rounds))

    return ServerApp(server_fn=server_fn)
