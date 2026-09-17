"""The aggregation hook preserves submission identity without changing legacy inputs."""

from types import SimpleNamespace

import numpy as np
import torch

from fltest.core import HookRunner
from fltest.core.config import RunSpec


def test_reference_before_aggregate_exposes_identified_submissions(monkeypatch):
    from fltest.frameworks.reference.adapter import ReferenceAdapter

    spec = RunSpec(
        run_id="identified", run_name="identified", framework="reference",
        num_clients=2, num_rounds=1, client_epochs=1, client_lr=0.01,
    )
    adapter = ReferenceAdapter()
    monkeypatch.setattr(
        adapter, "_new_model", lambda _spec: torch.nn.Sequential(
            torch.nn.Flatten(), torch.nn.Linear(1, 2)
        ),
    )
    batch = {"img": torch.ones(2, 1, 1, 1), "label": torch.tensor([0, 1])}
    data = {"c2loader": {0: [batch], 1: [batch]}, "test_loader": [batch]}
    seen = []
    runner = HookRunner()

    def inspect(ctx):
        seen.append(ctx)
        assert [item.client_id for item in ctx.client_submissions] == [0, 1]
        assert [item.num_samples for item in ctx.client_submissions] == [2, 2]
        assert len(ctx.global_state) == len(ctx.client_submissions[0].update)
        for record, (update, weight) in zip(ctx.client_submissions, ctx.updates_and_weights):
            assert weight == record.num_samples
            assert all(np.array_equal(a, b) for a, b in zip(record.update, update))

        # The old tuple-based input remains mutable and controls aggregation.
        ctx.updates_and_weights = [ctx.updates_and_weights[0]]

    runner.register("before_aggregate", inspect)
    result = adapter.run_simulation(spec, data, runner)

    assert result.status == "success"
    assert len(seen) == 1
    assert all(
        np.array_equal(actual, expected)
        for actual, expected in zip(seen[0].new_global_state,
                                    seen[0].client_submissions[0].update)
    )


def test_flower_before_aggregate_preserves_ids_and_tracks_global_model():
    from flwr.common import Code, ndarrays_to_parameters, parameters_to_ndarrays

    from fltest.frameworks.flower.server import HookedFedAvg

    spec = RunSpec(run_id="identified", run_name="identified", framework="flwr")
    initial = [np.array([1.0], dtype=np.float32)]
    runner = HookRunner()
    seen = []

    def inspect(ctx):
        seen.append(ctx)
        if ctx.round == 1:
            assert [item.client_id for item in ctx.client_submissions] == [7, 2]
            assert [item.num_samples for item in ctx.client_submissions] == [3, 1]
            assert np.array_equal(ctx.global_state[0], initial[0])
            assert len(ctx.updates_and_weights) == 2
            ctx.updates_and_weights = [ctx.updates_and_weights[0]]
        else:
            assert np.array_equal(ctx.global_state[0], np.array([3.0]))

    runner.register("before_aggregate", inspect)
    strategy = HookedFedAvg(
        hook_runner=runner, spec=spec, history={},
        initial_parameters=ndarrays_to_parameters(initial),
    )

    def response(client_id, value, weight):
        return (SimpleNamespace(), SimpleNamespace(
            status=SimpleNamespace(code=Code.OK),
            parameters=ndarrays_to_parameters([np.array([value], dtype=np.float32)]),
            num_examples=weight, metrics={"cid": client_id},
        ))

    first, _ = strategy.aggregate_fit(1, [response(7, 3.0, 3), response(2, 5.0, 1)], [])
    assert np.array_equal(parameters_to_ndarrays(first)[0], np.array([3.0]))
    strategy.aggregate_fit(2, [response(2, 4.0, 1), response(7, 6.0, 3)], [])
    assert [item.client_id for item in seen[1].client_submissions] == [2, 7]


def test_flower_keeps_legacy_aggregation_without_a_client_id():
    from flwr.common import Code, ndarrays_to_parameters, parameters_to_ndarrays

    from fltest.frameworks.flower.server import HookedFedAvg

    spec = RunSpec(run_id="identified", run_name="identified", framework="flwr")
    runner = HookRunner()
    seen = []
    runner.register("before_aggregate", lambda ctx: seen.extend(ctx.client_submissions))
    strategy = HookedFedAvg(hook_runner=runner, spec=spec, history={})
    result = SimpleNamespace(
        status=SimpleNamespace(code=Code.OK),
        parameters=ndarrays_to_parameters([np.array([2.0], dtype=np.float32)]),
        num_examples=1, metrics={},
    )

    aggregated, _ = strategy.aggregate_fit(1, [(SimpleNamespace(), result)], [])
    assert seen[0].client_id is None
    assert np.array_equal(parameters_to_ndarrays(aggregated)[0], np.array([2.0]))
