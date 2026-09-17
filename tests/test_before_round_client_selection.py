"""before_round selects who trains, before any client work is dispatched."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from fltest.core import HookRunner
from fltest.core.client_selection import resolve_selected_clients
from fltest.core.config import RunSpec


@pytest.mark.parametrize("selection", [[], [0, 0], [-1], [2], [True], {0}])
def test_invalid_client_selection_is_rejected(selection):
    with pytest.raises(ValueError, match="before_round selected_clients"):
        resolve_selected_clients(selection, 2)


def test_reference_before_round_controls_participation(monkeypatch):
    from fltest.frameworks.reference import adapter as reference_module

    spec = RunSpec(
        run_id="selection", run_name="selection", framework="reference",
        num_clients=3, num_rounds=2, client_epochs=1,
    )
    adapter = reference_module.ReferenceAdapter()
    monkeypatch.setattr(
        adapter, "_new_model", lambda _spec: torch.nn.Sequential(
            torch.nn.Flatten(), torch.nn.Linear(1, 2)
        ),
    )
    monkeypatch.setattr(reference_module, "train", lambda *args, **kwargs: None)
    batch = {"img": torch.ones(2, 1, 1, 1), "label": torch.tensor([0, 1])}
    data = {
        "c2loader": {cid: [batch] for cid in range(3)},
        "test_loader": [batch],
    }
    runner = HookRunner()
    trained = []
    submissions = []

    def select(ctx):
        assert ctx.selected_clients == (0, 1, 2)
        ctx.selected_clients = (2, 0) if ctx.round == 1 else (1,)

    runner.register("before_round", select)
    runner.register("before_client_train", lambda ctx: trained.append((ctx.round, ctx.client_id)))
    runner.register(
        "before_aggregate",
        lambda ctx: submissions.append(
            (ctx.round, tuple(item.client_id for item in ctx.client_submissions))
        ),
    )

    result = adapter.run_simulation(spec, data, runner)

    assert result.status == "success"
    assert trained == [(1, 2), (1, 0), (2, 1)]
    assert submissions == [(1, (2, 0)), (2, (1,))]


def test_flower_before_round_selects_partition_ids_before_dispatch():
    from flwr.common import Code, ndarrays_to_parameters

    from fltest.frameworks.flower.server import HookedFedAvg

    events = []

    class Proxy:
        def __init__(self, transport_id, partition_id):
            self.cid = transport_id
            self.partition_id = partition_id

        def get_properties(self, ins, timeout, group_id):
            events.append(("identify", self.partition_id))
            return SimpleNamespace(
                status=SimpleNamespace(code=Code.OK),
                properties={"cid": self.partition_id},
            )

    class Manager:
        def __init__(self):
            self.proxies = [Proxy("node-99", 1), Proxy("node-42", 2), Proxy("node-73", 0)]

        def num_available(self):
            return len(self.proxies)

        def sample(self, num_clients, min_num_clients):
            events.append(("sample", num_clients))
            assert num_clients == 3
            return self.proxies

    spec = RunSpec(run_id="selection", run_name="selection", framework="flwr", num_clients=3)
    runner = HookRunner()

    def select(ctx):
        events.append(("before_round", ctx.round))
        assert ctx.selected_clients == (0, 1, 2)
        assert np.array_equal(ctx.global_state[0], np.array([1.0]))
        ctx.selected_clients = (2, 0) if ctx.round == 1 else (1,)

    runner.register("before_round", select)
    parameters = ndarrays_to_parameters([np.array([1.0], dtype=np.float32)])
    strategy = HookedFedAvg(
        hook_runner=runner, spec=spec, history={}, initial_parameters=parameters,
        fraction_fit=1.0, min_fit_clients=3, min_available_clients=3,
    )
    manager = Manager()

    first = strategy.configure_fit(1, parameters, manager)
    second = strategy.configure_fit(2, parameters, manager)

    assert [proxy.partition_id for proxy, _ in first] == [2, 0]
    assert [proxy.partition_id for proxy, _ in second] == [1]
    assert events[0] == ("before_round", 1)
    assert events.count(("before_round", 1)) == 1
    assert events.count(("before_round", 2)) == 1
    assert [event for event in events if event[0] == "identify"] == [
        ("identify", 1), ("identify", 2), ("identify", 0)
    ]


def test_flower_full_participation_needs_no_identity_probe():
    from flwr.common import ndarrays_to_parameters

    from fltest.frameworks.flower.server import HookedFedAvg

    class Manager:
        def num_available(self):
            return 2

        def sample(self, num_clients, min_num_clients):
            return [SimpleNamespace(cid="node-a"), SimpleNamespace(cid="node-b")]

    spec = RunSpec(run_id="all", run_name="all", framework="flwr", num_clients=2)
    runner = HookRunner()
    seen = []
    runner.register("before_round", lambda ctx: seen.append(ctx.selected_clients))
    parameters = ndarrays_to_parameters([np.array([1.0], dtype=np.float32)])
    strategy = HookedFedAvg(
        hook_runner=runner, spec=spec, history={}, initial_parameters=parameters,
        fraction_fit=1.0, min_fit_clients=2, min_available_clients=2,
    )

    configured = strategy.configure_fit(1, parameters, Manager())

    assert seen == [(0, 1)]
    assert len(configured) == 2


def test_flower_selective_round_requires_reported_partition_ids():
    from flwr.common import Code, ndarrays_to_parameters

    from fltest.frameworks.flower.server import HookedFedAvg

    class Proxy:
        def __init__(self, transport_id):
            self.cid = transport_id

        def get_properties(self, ins, timeout, group_id):
            return SimpleNamespace(status=SimpleNamespace(code=Code.OK), properties={})

    class Manager:
        def num_available(self):
            return 2

        def sample(self, num_clients, min_num_clients):
            return [Proxy("transport-id"), Proxy("other")]

    spec = RunSpec(run_id="missing", run_name="missing", framework="flwr", num_clients=2)
    runner = HookRunner()
    runner.register("before_round", lambda ctx: setattr(ctx, "selected_clients", (0,)))
    parameters = ndarrays_to_parameters([np.array([1.0], dtype=np.float32)])
    strategy = HookedFedAvg(
        hook_runner=runner, spec=spec, history={}, initial_parameters=parameters,
        fraction_fit=1.0, min_fit_clients=2, min_available_clients=2,
    )

    with pytest.raises(ValueError, match="valid partition ID"):
        strategy.configure_fit(1, parameters, Manager())


def test_flower_rejects_empty_selection_before_sampling():
    from flwr.common import ndarrays_to_parameters

    from fltest.frameworks.flower.server import HookedFedAvg

    class Manager:
        def num_available(self):
            raise AssertionError("client manager should not be consulted")

    spec = RunSpec(run_id="empty", run_name="empty", framework="flwr", num_clients=2)
    runner = HookRunner()
    runner.register("before_round", lambda ctx: setattr(ctx, "selected_clients", []))
    parameters = ndarrays_to_parameters([np.array([1.0], dtype=np.float32)])
    strategy = HookedFedAvg(hook_runner=runner, spec=spec, history={})

    with pytest.raises(ValueError, match="nonempty"):
        strategy.configure_fit(1, parameters, Manager())


def test_flower_client_reports_partition_id_before_fit():
    from flwr.common import Code, GetPropertiesIns

    from fltest.frameworks.flower.client import FlowerClient

    client = object.__new__(FlowerClient)
    client.cid = 7
    assert client.get_properties({}) == {"cid": 7}
    response = client.to_client().get_properties(GetPropertiesIns({}))
    assert response.status.code == Code.OK
    assert response.properties == {"cid": 7}
