from flwr.simulation import run_simulation

from fl_testing.frameworks.utils import (
    seed_every_thing,
    get_final_round_results,
    test_case_own_gm_model_summation,
)
from fl_testing.frameworks.models import (
    get_pytorch_model,
    sum_model_weights_pytorch,
    test,
)
from fl_testing.frameworks.pytorch_fl_dataset import get_dataset_for_framework

from fltest.core import HookContext, HookRunner
from fltest.adapters.flower.client import get_client_app
from fltest.adapters.flower.utils import set_parameters
from fltest.adapters.flower.server import get_server_app


def run_flower_simulation(cfg, hook_runner: HookRunner):
    seed_every_thing(cfg.seed)

    final_round_loss = -1
    final_round_accuracy = -1
    sum_of_weights = -1
    own_implmentation_sum_of_weights = -1
    history = {}

    # Hook: before_simulation
    ctx_init = HookContext(cfg=cfg, history=history)
    hook_runner.run("before_simulation", ctx_init)

    num_gpus_total = 1 if cfg.device == "cuda" else cfg.total_gpus
    num_gpus_per_client = (1.0 / cfg.num_clients) if cfg.device == "cuda" else 0.0
    init_args = {"num_cpus": cfg.total_cpus, "num_gpus": num_gpus_total}
    backend_config = {
        "client_resources": {"num_cpus": 1, "num_gpus": num_gpus_per_client},
        "init_args": init_args,
    }

    def _central_evaluate(server_round, parameters, config):
        nonlocal final_round_loss, final_round_accuracy, sum_of_weights
        nonlocal own_implmentation_sum_of_weights, history
        net = get_pytorch_model(
            cfg.model_name,
            cfg.model_cache_path,
            deterministic=cfg.deterministic,
            channels=cfg.channels,
            seed=cfg.seed,
        )
        net.to(cfg.device)
        set_parameters(net, parameters)
        loss, accuracy = test(
            net,
            test_data_loader,
            device=cfg.device,
            loss_fn=cfg.loss_fn,
            seed=cfg.seed,
        )
        print(
            f"Server-side evaluation loss {loss} / accuracy {accuracy} in round {server_round}"
        )
        final_round_loss = loss
        final_round_accuracy = accuracy
        sum_of_weights = sum_model_weights_pytorch(net)
        if server_round > 0:
            own_implmentation_sum_of_weights = test_case_own_gm_model_summation(cfg)

        # Hook: after_round
        ctx_round = HookContext(
            cfg=cfg,
            round=server_round,
            global_state=parameters,
            test_data=test_data_loader,
            metrics={"loss": loss, "accuracy": accuracy},
            history=history,
        )
        if history is not None:
            history[server_round] = ctx_round.metrics
        hook_runner.run("after_round", ctx_round)

        return loss, {"accuracy": accuracy}

    fl_dataset_dict = get_dataset_for_framework(cfg)
    test_data_loader = fl_dataset_dict["test_data"]
    c2data = fl_dataset_dict["c2data"]

    # Hook: on_data_distribute (plugins can mutate dist_dict)
    ctx_data = HookContext(
        cfg=cfg,
        dist_dict=c2data,
        test_data=test_data_loader,
    )
    hook_runner.run("on_data_distribute", ctx_data)
    c2data = ctx_data.dist_dict if ctx_data.dist_dict is not None else c2data

    # Hooked server: lets server-side hooks (@before_aggregate, @on_aggregate,
    # @after_aggregate) actually take effect. aggregate_fit unconditionally
    # rebuilds parameters from ctx.new_global_state, so with no defense hook
    # the result is the FedAvg round-trip (identical values).
    # hook_runner=None on the client side so workers re-import FLTEST_HOOKS in
    # each Ray worker.
    server_app = get_server_app(cfg, central_eval_fn=_central_evaluate, hook_runner=hook_runner)
    client_app = get_client_app(cfg, c2data_loader=c2data)

    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=cfg.num_clients,
        backend_config=backend_config,
    )

    # Hook: after_simulation
    ctx_final = HookContext(
        cfg=cfg,
        history=history,
        metrics={"loss": final_round_loss, "accuracy": final_round_accuracy},
    )
    hook_runner.run("after_simulation", ctx_final)

    result = get_final_round_results(
        final_round_loss,
        final_round_accuracy,
        pytorch_gm_sum=own_implmentation_sum_of_weights,
        framework_gm_sum=sum_of_weights,
    )
    return result
