from diskcache import Index
from flwr.client import ClientApp
from flwr.client import NumPyClient

from fl_testing.frameworks.models import (
    get_pytorch_model,
    sum_model_weights_pytorch,
    train,
    test,
)
from fl_testing.frameworks.utils import seed_every_thing

from fltest.adapters.flower.utils import set_parameters, get_parameters
from fltest.core import HookContext, HookRunner


class FlowerClient(NumPyClient):
    def __init__(self, client_data, cfg, cid, hook_runner: HookRunner):
        seed_every_thing(cfg.seed)
        self.net = get_pytorch_model(
            cfg.model_name,
            model_cache_dir=cfg.model_cache_path,
            deterministic=cfg.deterministic,
            channels=cfg.channels,
            seed=cfg.seed,
        ).to(cfg.device)
        self.trainloader = client_data
        self.valloader = client_data
        self.cfg = cfg
        self.cid = cid
        self.hook_runner = hook_runner

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        before_trining_ws = sum_model_weights_pytorch(self.net)

        # Hook: before_client_train (plugins can mutate client_data / loader)
        ctx_pre = HookContext(
            cfg=self.cfg,
            round=config.get("server_round", 0),
            client_id=self.cid,
            client_data=self.trainloader,
            global_state=parameters,  # Expose global params for gradient-based hooks (e.g. DLG)
        )
        self.hook_runner.run("before_client_train", ctx_pre)
        if ctx_pre.client_data is not None:
            self.trainloader = ctx_pre.client_data

        train(
            self.net,
            self.trainloader,
            epochs=self.cfg.client_epochs,
            device=self.cfg.device,
            loss_fn=self.cfg.loss_fn,
            opitmzer_name=self.cfg.optimizer,
            seed=self.cfg.seed,
        )
        after_trining_ws = sum_model_weights_pytorch(self.net)

        # Use dataset length (number of examples), not loader length (number of
        # batches). Flower uses num_samples as a FedAvg weight; per-batch
        # weighting silently distorts aggregation under unequal partitions.
        num_samples = (
            len(self.trainloader.dataset)
            if hasattr(self.trainloader, "dataset")
            else len(self.trainloader)
        )

        temp_cache = Index(self.cfg.fw_cache_path)
        temp_cache[f"cid_{self.cid}"] = self.net.state_dict(), num_samples

        parameters_out = get_parameters(self.net)

        # Hook: after_client_train (plugins can mutate the update)
        ctx = HookContext(
            cfg=self.cfg,
            round=config.get("server_round", 0),
            client_id=self.cid,
            client_update=parameters_out,
            num_samples=num_samples,
        )
        self.hook_runner.run("after_client_train", ctx)
        parameters_out = ctx.client_update if ctx.client_update is not None else parameters_out

        return (
            parameters_out,
            num_samples,
            {
                "cid": self.cid,
                "before_train": before_trining_ws,
                "after_train": after_trining_ws,
            },
        )

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, accuracy = test(
            self.net,
            self.valloader,
            self.cfg.device,
            loss_fn=self.cfg.loss_fn,
        )
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}


def get_client_app(cfg, c2data_loader, hook_runner: HookRunner = None):
    """Build ClientApp. If hook_runner is None, build it inside the worker so FLTEST_HOOKS (e.g. examples/hooks) run there and can write tmp/ etc."""
    def client_fn(context):
        from fltest.core import HookRunner, hooks
        if hook_runner is None:
            hooks.import_convention_hooks()
            runner = HookRunner()
            hooks.apply_to(runner)
        else:
            runner = hook_runner
        partition_id = context.node_config["partition-id"]
        client_data = c2data_loader[partition_id]
        return FlowerClient(
            client_data, cfg, cid=partition_id, hook_runner=runner
        ).to_client()

    client_app = ClientApp(client_fn=client_fn)
    return client_app
