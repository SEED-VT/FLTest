from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg

from fl_testing.frameworks.models import get_pytorch_model
from fl_testing.frameworks.utils import seed_every_thing

from fltest.adapters.flower.utils import get_parameters
from fltest.core import HookContext, HookRunner


def weighted_average(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}


def _fit_metrics_aggregation_fn(metrics):
    return {"loss temp": 0.0, "accuracy-temp": 0.0}


class HookedFedAvg(FedAvg):
    """FedAvg strategy that runs before_round, before_aggregate, on_aggregate, after_aggregate hooks."""

    def __init__(self, hook_runner: HookRunner, cfg, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._hook_runner = hook_runner
        self._cfg = cfg

    def aggregate_fit(
        self,
        server_round: int,
        results,  # List[Tuple[ClientProxy, FitRes]]
        failures,
    ):
        ctx_round = HookContext(cfg=self._cfg, round=server_round)
        self._hook_runner.run("before_round", ctx_round)

        # Mirror Flower's own FedAvg behavior: bail only when there are no
        # results at all (failures are tracked separately by FedAvg).
        if not results:
            return None, {}

        updates_as_ndarrays = [
            (parameters_to_ndarrays(res.parameters), res.num_examples)
            for _proxy, res in results
        ]
        ctx = HookContext(
            cfg=self._cfg,
            round=server_round,
            updates_and_weights=updates_as_ndarrays,
            global_state=None,
        )
        self._hook_runner.run("before_aggregate", ctx)

        parameters_aggregated, metrics_aggregated = super().aggregate_fit(
            server_round, results, failures
        )

        if parameters_aggregated is not None:
            ctx.new_global_state = parameters_to_ndarrays(parameters_aggregated)
        self._hook_runner.run("on_aggregate", ctx)
        self._hook_runner.run("after_aggregate", ctx)

        # Unconditionally rebuild aggregated parameters from ctx.new_global_state.
        # With no defense hook this is a value-preserving round trip. With a
        # defense hook (Krum, median, ...) that overwrote new_global_state, this
        # is how the override takes effect.
        if ctx.new_global_state is not None:
            parameters_aggregated = ndarrays_to_parameters(ctx.new_global_state)

        return parameters_aggregated, metrics_aggregated


def get_server_app(cfg, central_eval_fn, hook_runner: HookRunner):
    seed_every_thing(cfg.seed)
    net2 = get_pytorch_model(
        cfg.model_name,
        cfg.model_cache_path,
        deterministic=cfg.deterministic,
        channels=cfg.channels,
        seed=cfg.seed,
    )
    initial_parameters = get_parameters(net2)

    def _fit_config(server_round: int):
        return {"server_round": server_round}

    def server_fn(context):
        strategy = HookedFedAvg(
            hook_runner=hook_runner,
            cfg=cfg,
            fraction_fit=1.0,
            fraction_evaluate=0.0,
            min_fit_clients=cfg.num_clients,
            min_evaluate_clients=0,
            min_available_clients=cfg.num_clients,
            evaluate_metrics_aggregation_fn=weighted_average,
            evaluate_fn=central_eval_fn,
            fit_metrics_aggregation_fn=_fit_metrics_aggregation_fn,
            on_fit_config_fn=_fit_config,
            initial_parameters=ndarrays_to_parameters(initial_parameters),
        )
        config = ServerConfig(num_rounds=cfg.num_rounds)
        return ServerAppComponents(strategy=strategy, config=config)

    server_app = ServerApp(server_fn=server_fn)
    return server_app
