"""Flower NumPyClient that emits FLTest client-side hooks inside Ray workers.

The hook runner is rebuilt *in the worker* from the (picklable) ``RunSpec`` via
``build_hook_runner`` — closures are never shipped across processes. Metrics that
client-side hooks record (e.g. DLG reconstruction quality) are returned in the fit-metrics
dict so the server can surface them, since worker-side context does not propagate to the
driver otherwise.
"""

from __future__ import annotations

from flwr.client import ClientApp, NumPyClient

from fltest.core import HookContext
from fltest.core.config import RunSpec
from fltest.core.wiring import build_hook_runner
from fltest.data.models import get_model, model_weight_sum, test, train
from fltest.frameworks.flower.utils import get_parameters, numeric_metrics, set_parameters


class FlowerClient(NumPyClient):
    def __init__(self, client_data, spec: RunSpec, cid: int, hook_runner):
        self.spec = spec
        self.cid = cid
        self.hook_runner = hook_runner
        self.trainloader = client_data
        self.net = get_model(
            spec.model_name, spec.model_cache_path, channels=spec.channels,
            num_classes=spec.num_classes, deterministic=spec.deterministic,
        ).to(spec.device)

    def fit(self, parameters, config):
        spec = self.spec
        set_parameters(self.net, parameters)
        rnd = int(config.get("server_round", 0))

        ctx_pre = HookContext(
            cfg=spec, framework="flwr", run_name=spec.run_name, round=rnd, client_id=self.cid,
            client_data=self.trainloader, global_state=parameters,
        )
        self.hook_runner.run("before_client_train", ctx_pre)
        loader = ctx_pre.client_data if ctx_pre.client_data is not None else self.trainloader

        train(
            self.net, loader, epochs=spec.client_epochs, device=spec.device,
            loss_fn=spec.loss_fn, optimizer_name=spec.optimizer, lr=spec.client_lr,
        )
        update = get_parameters(self.net)
        n_samples = sum(b["label"].size(0) for b in loader)

        ctx_post = HookContext(
            cfg=spec, framework="flwr", run_name=spec.run_name, round=rnd, client_id=self.cid,
            client_update=update, num_samples=n_samples, global_state=parameters, model=self.net,
        )
        self.hook_runner.run("after_client_train", ctx_post)
        update = ctx_post.client_update if ctx_post.client_update is not None else update

        metrics = {"cid": self.cid}
        metrics.update(numeric_metrics(ctx_pre.metrics))
        metrics.update(numeric_metrics(ctx_post.metrics))
        return update, n_samples, metrics

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, acc = test(self.net, self.trainloader, self.spec.device, loss_fn=self.spec.loss_fn)
        return float(loss), len(self.trainloader), {"accuracy": float(acc)}


def get_client_app(spec: RunSpec, c2loader) -> ClientApp:
    """Build the Flower ClientApp; hooks are reconstructed per-worker from ``spec``."""

    def client_fn(context):
        runner = build_hook_runner(spec)  # rebuilt in-worker from the picklable spec
        partition_id = context.node_config["partition-id"]
        return FlowerClient(c2loader[partition_id], spec, partition_id, runner).to_client()

    return ClientApp(client_fn=client_fn)
