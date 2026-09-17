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

from flwr.common import Code, GetPropertiesIns, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

from fltest.core import ClientSubmission, HookContext, HookRunner
from fltest.core.client_selection import resolve_selected_clients
from fltest.core.config import RunSpec
from fltest.data.models import get_model, model_weight_sum, test
from fltest.data.utils import aggregate_ndarrays
from fltest.frameworks.flower.utils import get_parameters, set_parameters


def fit_config(server_round: int) -> Dict[str, int]:
    """Send the lifecycle round to clients so their hook contexts match the server."""
    return {"server_round": server_round}


class HookedFedAvg(FedAvg):
    def __init__(self, hook_runner: HookRunner, spec: RunSpec, history: Dict, **kwargs):
        initial_parameters = kwargs.get("initial_parameters")
        super().__init__(**kwargs)
        self._hooks = hook_runner
        self._spec = spec
        self._history = history
        self._last_fit_metrics: Dict[str, float] = {}
        self._last_server_metrics: Dict = {}
        self._client_id_by_proxy: Dict[str, int] = {}
        self._global_state = (
            parameters_to_ndarrays(initial_parameters) if initial_parameters is not None else None
        )

    def configure_fit(self, server_round, parameters, client_manager):
        """Run before_round before dispatch, then retain only selected FLTest clients."""
        ctx = HookContext(
            cfg=self._spec, framework="flwr", run_name=self._spec.run_name,
            round=server_round, global_state=parameters_to_ndarrays(parameters),
            selected_clients=tuple(range(self._spec.num_clients)), history=self._history,
        )
        self._hooks.run("before_round", ctx)
        selected = resolve_selected_clients(ctx.selected_clients, self._spec.num_clients)

        configured = super().configure_fit(server_round, parameters, client_manager)
        if selected == tuple(range(self._spec.num_clients)) or not configured:
            return configured

        by_client_id = {}
        for proxy, fit_ins in configured:
            client_id = self._client_id_by_proxy.get(proxy.cid)
            if client_id is None:
                response = proxy.get_properties(
                    GetPropertiesIns({}), timeout=None, group_id=server_round
                )
                client_id = response.properties.get("cid") if response.status.code == Code.OK else None
                if (
                    not isinstance(client_id, int) or isinstance(client_id, bool)
                    or not 0 <= client_id < self._spec.num_clients
                ):
                    raise ValueError(
                        "Selective before_round requires every Flower client to report "
                        "a valid partition ID in get_properties"
                    )
                self._client_id_by_proxy[proxy.cid] = client_id
            if client_id in by_client_id:
                raise ValueError(f"Flower clients report duplicate partition ID {client_id}")
            by_client_id[client_id] = (proxy, fit_ins)

        missing = [cid for cid in selected if cid not in by_client_id]
        if missing:
            raise ValueError(f"Selected Flower clients are unavailable: {missing}")
        return [by_client_id[cid] for cid in selected]

    def aggregate_fit(self, server_round, results, failures):
        successful = [(p, r) for p, r in results if r.status.code == Code.OK]
        if not successful:
            return None, {}

        updates_and_weights = []
        client_submissions = []
        for _, response in successful:
            client_id = response.metrics.get("cid")
            if not isinstance(client_id, int) or isinstance(client_id, bool):
                # Custom Flower clients may not report FLTest's partition ID. Preserve
                # their existing aggregation behavior without inventing an identity.
                client_id = None
            update = parameters_to_ndarrays(response.parameters)
            updates_and_weights.append((update, response.num_examples))
            client_submissions.append(
                ClientSubmission(client_id, tuple(update), response.num_examples)
            )

        ctx = HookContext(
            cfg=self._spec, framework="flwr", run_name=self._spec.run_name, round=server_round,
            updates_and_weights=updates_and_weights,
            client_submissions=tuple(client_submissions), global_state=self._global_state,
        )
        self._hooks.run("before_aggregate", ctx)
        uw = ctx.updates_and_weights if ctx.updates_and_weights is not None else updates_and_weights

        aggregated = aggregate_ndarrays(uw)
        ctx.new_global_state = aggregated
        self._hooks.run("on_aggregate", ctx)
        aggregated = ctx.new_global_state if ctx.new_global_state is not None else aggregated
        ctx.new_global_state = aggregated
        self._global_state = aggregated
        self._hooks.run("after_aggregate", ctx)
        self._last_server_metrics = dict(ctx.metrics)

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
            ctx.metrics.update(strat_metrics._last_server_metrics)
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
            evaluate_fn=evaluate_fn, on_fit_config_fn=fit_config,
            initial_parameters=initial_parameters,
        )
        evaluate_fn._strategy = strategy  # let evaluate read the round's fit metrics
        return ServerAppComponents(strategy=strategy, config=ServerConfig(num_rounds=spec.num_rounds))

    return ServerApp(server_fn=server_fn)
