"""Online FLDetector's hook contract and aggregation composition."""

import numpy as np
import pytest

from fltest.core import ClientSubmission, HookContext
from fltest.core.config import DefenseSpec, RunSpec
from fltest.core.wiring import build_hook_runner
from fltest.defenses.fldetector import FLDetectorDefense, _hessian_product, _suspected_ids


def _spec(**kwargs):
    return RunSpec(**{"run_id": "det", "run_name": "det", "framework": "reference", **kwargs})


def _round_context(spec, round_number, model, gradients):
    submissions = tuple(
        ClientSubmission(cid, (np.array([model - gradient], dtype=np.float32),), 1)
        for cid, gradient in gradients.items()
    )
    return HookContext(
        cfg=spec, round=round_number,
        global_state=[np.array([model], dtype=np.float32)],
        client_submissions=submissions,
        updates_and_weights=[(list(item.update), item.num_samples) for item in submissions],
    )


def test_hessian_product_uses_gradient_curvature():
    s = np.array([2.0, 1.0])
    y = np.array([4.0, 2.0])
    assert np.allclose(_hessian_product([(s, y)], np.array([3.0, -1.0])), [6.0, -2.0])
    assert np.array_equal(
        _hessian_product([(np.zeros(2), np.zeros(2))], np.array([3.0, -1.0])),
        np.zeros(2),
    )


def test_gap_statistics_separates_obvious_high_score_cluster():
    scores = {0: 0.01, 1: 0.011, 2: 0.012, 3: 0.91, 4: 0.93}
    assert _suspected_ids(scores, max_clusters=3, samples=20, seed=1) == {3, 4}
    assert _suspected_ids({cid: 0.2 for cid in scores}, 3, 20, 1) == set()


def test_fldetector_filters_current_round_then_excludes_future_training():
    spec = _spec(num_clients=5)
    detector = FLDetectorDefense(window_size=1, start_round=3, max_clusters=3)
    detector.before_simulation(HookContext(cfg=spec))
    for rnd, model in ((1, 0.0), (2, -1.0), (3, -2.0)):
        gradients = {0: 1.0, 1: 1.01, 2: 1.02, 3: 1.03, 4: 1.04}
        if rnd == 3:
            gradients[4] = 10.0
        ctx = _round_context(spec, rnd, model, gradients)
        detector.before_aggregate(ctx)
        if rnd < 3:
            assert len(ctx.updates_and_weights) == 5
        else:
            assert detector.excluded_clients == {4}
            assert len(ctx.updates_and_weights) == 4
            assert ctx.metrics["fldetector_detected_clients"] == [4]
        ctx.new_global_state = [np.array([model - 1.0], dtype=np.float32)]
        detector.after_aggregate(ctx)

    next_round = HookContext(cfg=spec, selected_clients=tuple(range(5)))
    detector.before_round(next_round)
    assert next_round.selected_clients == (0, 1, 2, 3)


def test_fldetector_requires_identity_and_unmodified_alignment():
    spec = _spec()
    detector = FLDetectorDefense()
    ctx = _round_context(spec, 1, 0.0, {0: 1.0, 1: 1.1})
    ctx.client_submissions = (ClientSubmission(None, ctx.client_submissions[0].update, 1),
                              ctx.client_submissions[1])
    with pytest.raises(ValueError, match="unique, stable client IDs"):
        detector.before_aggregate(ctx)

    ctx = _round_context(spec, 1, 0.0, {0: 1.0, 1: 1.1})
    ctx.updates_and_weights.reverse()
    with pytest.raises(ValueError, match="aligned"):
        detector.before_aggregate(ctx)


def test_fldetector_must_run_before_optional_robust_aggregator():
    spec = _spec(defenses=[DefenseSpec(name="median"), DefenseSpec(name="fldetector")])
    with pytest.raises(ValueError, match="must precede"):
        build_hook_runner(spec, load_env_hooks=False)

    spec.defenses.reverse()
    runner = build_hook_runner(spec, load_env_hooks=False)
    handlers = runner._registry["before_aggregate"]
    assert [type(handler.__self__).__name__ for handler in handlers] == [
        "FLDetectorDefense", "MedianDefense",
    ]


def test_flower_aggregation_uses_fldetector_filter_and_keeps_hook_metrics():
    from types import SimpleNamespace

    from flwr.common import Code, ndarrays_to_parameters, parameters_to_ndarrays

    from fltest.core import HookRunner
    from fltest.frameworks.flower.server import HookedFedAvg

    spec = _spec(framework="flwr", num_clients=5)
    runner = HookRunner()
    detector = FLDetectorDefense(window_size=1, start_round=3, max_clusters=3)
    detector.attach(runner)
    strategy = HookedFedAvg(
        hook_runner=runner, spec=spec, history={},
        initial_parameters=ndarrays_to_parameters([np.array([0.0], dtype=np.float32)]),
    )

    for rnd, model in ((1, 0.0), (2, -1.0), (3, -2.0)):
        gradients = [1.0, 1.01, 1.02, 1.03, 1.04 if rnd < 3 else 10.0]
        results = [(
            SimpleNamespace(),
            SimpleNamespace(
                status=SimpleNamespace(code=Code.OK),
                parameters=ndarrays_to_parameters([
                    np.array([model - gradient], dtype=np.float32)
                ]),
                num_examples=1,
                metrics={"cid": cid},
            ),
        ) for cid, gradient in enumerate(gradients)]
        aggregated, _ = strategy.aggregate_fit(rnd, results, [])
        if rnd == 3:
            assert np.allclose(parameters_to_ndarrays(aggregated)[0], [model - 1.015])
            assert strategy._last_server_metrics["fldetector_detected_clients"] == [4]
        # Keep the synthetic model trajectory controlled for the next prediction.
        strategy._global_state = [np.array([model - 1.0], dtype=np.float32)]
