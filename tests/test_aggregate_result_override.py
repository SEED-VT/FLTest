"""The on_aggregate hook can replace the model used by later phases and rounds."""

from types import SimpleNamespace

import numpy as np
import torch

from fltest.core import HookRunner
from fltest.core.config import RunSpec


def test_reference_on_aggregate_replaces_global_model(monkeypatch):
    from fltest.frameworks.reference import adapter as reference_module

    spec = RunSpec(
        run_id="override", run_name="override", framework="reference",
        num_clients=1, num_rounds=2, client_epochs=1,
    )
    adapter = reference_module.ReferenceAdapter()
    monkeypatch.setattr(
        adapter, "_new_model", lambda _spec: torch.nn.Sequential(
            torch.nn.Flatten(), torch.nn.Linear(1, 2)
        ),
    )
    monkeypatch.setattr(reference_module, "train", lambda *args, **kwargs: None)
    batch = {"img": torch.ones(2, 1, 1, 1), "label": torch.tensor([0, 1])}
    data = {"c2loader": {0: [batch]}, "test_loader": [batch]}
    runner = HookRunner()
    seen = []

    def replace(ctx):
        assert ctx.new_global_state is not None
        ctx.new_global_state = [np.zeros_like(part) for part in ctx.new_global_state]

    def observe(ctx):
        seen.append((ctx.round, [part.copy() for part in ctx.new_global_state]))

    def check_next_round(ctx):
        if ctx.round == 2:
            assert all(not np.any(part) for part in ctx.global_state)

    runner.register("on_aggregate", replace)
    runner.register("after_aggregate", observe)
    runner.register("before_round", check_next_round)
    result = adapter.run_simulation(spec, data, runner)

    assert result.status == "success"
    assert [round_number for round_number, _ in seen] == [1, 2]
    assert all(not np.any(part) for _, state in seen for part in state)
    assert result.final["gm_weight_sum"] == 0.0


def test_flower_on_aggregate_replaces_returned_and_next_round_model():
    from flwr.common import Code, ndarrays_to_parameters, parameters_to_ndarrays

    from fltest.frameworks.flower.server import HookedFedAvg

    spec = RunSpec(run_id="override", run_name="override", framework="flwr")
    initial = [np.array([1.0], dtype=np.float32)]
    runner = HookRunner()
    seen = []

    def replace(ctx):
        seen.append((ctx.round, "on", ctx.new_global_state[0].copy()))
        ctx.new_global_state = (
            [np.array([9.0], dtype=np.float32)] if ctx.round == 1 else None
        )

    def observe(ctx):
        seen.append((ctx.round, "after", ctx.new_global_state[0].copy()))

    def check_next_round(ctx):
        if ctx.round == 2:
            assert np.array_equal(ctx.global_state[0], np.array([9.0]))

    runner.register("on_aggregate", replace)
    runner.register("after_aggregate", observe)
    runner.register("before_aggregate", check_next_round)
    strategy = HookedFedAvg(
        hook_runner=runner, spec=spec, history={},
        initial_parameters=ndarrays_to_parameters(initial),
    )

    def response(value):
        return (SimpleNamespace(), SimpleNamespace(
            status=SimpleNamespace(code=Code.OK),
            parameters=ndarrays_to_parameters([np.array([value], dtype=np.float32)]),
            num_examples=1, metrics={"cid": 0},
        ))

    first, _ = strategy.aggregate_fit(1, [response(3.0)], [])
    second, _ = strategy.aggregate_fit(2, [response(4.0)], [])

    assert np.array_equal(parameters_to_ndarrays(first)[0], np.array([9.0]))
    assert np.array_equal(parameters_to_ndarrays(second)[0], np.array([4.0]))
    assert [(round_number, phase) for round_number, phase, _ in seen] == [
        (1, "on"), (1, "after"), (2, "on"), (2, "after")
    ]
    assert np.array_equal(seen[0][2], np.array([3.0]))
    assert np.array_equal(seen[1][2], np.array([9.0]))
    assert np.array_equal(seen[3][2], np.array([4.0]))
