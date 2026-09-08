"""Unit tests for attacks/defenses operating on the canonical ndarray update."""

import numpy as np
import torch

from fltest.core.hook_context import HookContext
from fltest.core.config import RunSpec


def _spec(**kw):
    base = dict(run_id="t", run_name="t", framework="reference", num_classes=10)
    base.update(kw)
    return RunSpec(**base)


def test_label_flip_relabels_batches():
    from fltest.attacks.label_flip import LabelFlipAttack

    loader = [{"img": torch.zeros(4, 1, 8, 8), "label": torch.tensor([0, 1, 2, 3])}]
    ctx = HookContext(cfg=_spec(), client_id=0, client_data=loader)
    LabelFlipAttack(shift=1).before_client_train(ctx)
    out = next(iter(ctx.client_data))
    assert out["label"].tolist() == [1, 2, 3, 4]


def test_sign_flip_reflects_update():
    from fltest.attacks.sign_flip import SignFlipAttack

    g = [np.ones((3,), dtype=np.float32)]
    u = [np.array([2.0, 2.0, 2.0], dtype=np.float32)]  # delta = +1
    ctx = HookContext(cfg=_spec(), client_id=0, client_update=u, global_state=g)
    SignFlipAttack(scale=1.0).after_client_train(ctx)
    # u' = g - (u - g) = 2g - u = 0
    assert np.allclose(ctx.client_update[0], 0.0)


def test_model_replacement_scales_local_delta():
    from fltest.attacks.model_replacement import ModelReplacementAttack

    g = [np.ones((3,), dtype=np.float32)]
    u = [np.full((3,), 2.0, dtype=np.float32)]
    ctx = HookContext(cfg=_spec(), round=4, client_id=1, client_update=u, global_state=g)

    ModelReplacementAttack(
        scale=5.0, target_round=4, target_clients=[1]
    ).after_client_train(ctx)

    assert np.allclose(ctx.client_update[0], 6.0)
    assert ctx.client_update[0].shape == (3,)
    assert ctx.client_update[0].dtype == np.float32
    assert ctx.metrics["model_replacement_scale"] == 5.0


def test_model_replacement_obeys_target_client_and_round():
    from fltest.attacks.model_replacement import ModelReplacementAttack

    g = [np.zeros((2,), dtype=np.float32)]
    original = [np.ones((2,), dtype=np.float32)]
    attack = ModelReplacementAttack(scale=10.0, target_round=3, target_clients=[1])

    wrong_client = HookContext(
        cfg=_spec(), round=3, client_id=0,
        client_update=[original[0].copy()], global_state=g,
    )
    attack.after_client_train(wrong_client)
    assert np.array_equal(wrong_client.client_update[0], original[0])

    wrong_round = HookContext(
        cfg=_spec(), round=2, client_id=1,
        client_update=[original[0].copy()], global_state=g,
    )
    attack.after_client_train(wrong_round)
    assert np.array_equal(wrong_round.client_update[0], original[0])


def test_model_replacement_auto_scale_is_shared_by_attackers():
    from fltest.attacks.model_replacement import ModelReplacementAttack

    g = [np.ones((1,), dtype=np.float32)]
    u = [np.full((1,), 2.0, dtype=np.float32)]
    ctx = HookContext(
        cfg=_spec(num_clients=6), round=1, client_id=4,
        client_update=u, global_state=g,
    )

    ModelReplacementAttack(target_clients=[1, 4]).after_client_train(ctx)

    # Two colluding attackers divide the six-client replacement factor: 6 / 2 = 3.
    assert np.allclose(ctx.client_update[0], 4.0)
    assert ctx.metrics["model_replacement_scale"] == 3.0


def test_model_replacement_reaches_target_under_equal_weight_fedavg():
    from fltest.attacks.model_replacement import ModelReplacementAttack
    from fltest.data.utils import aggregate_ndarrays

    g = [np.zeros((2,), dtype=np.float32)]
    target = [np.array([2.0, -3.0], dtype=np.float32)]
    ctx = HookContext(
        cfg=_spec(num_clients=4), round=1, client_id=0,
        client_update=target, global_state=g,
    )
    ModelReplacementAttack(target_clients=[0]).after_client_train(ctx)

    benign = [g[0].copy()]
    aggregated = aggregate_ndarrays([
        (ctx.client_update, 1), (benign, 1), (benign, 1), (benign, 1),
    ])
    assert np.allclose(aggregated[0], target[0])


def test_gradient_noise_clips_delta_norm():
    from fltest.defenses.gradient_noise import GradientNoiseDefense

    g = [np.zeros((100,), dtype=np.float32)]
    u = [np.full((100,), 10.0, dtype=np.float32)]  # delta norm = 100
    ctx = HookContext(cfg=_spec(), client_id=0, client_update=u, global_state=g)
    GradientNoiseDefense(clip_norm=1.0, sigma=0.0).after_client_train(ctx)
    delta_norm = float(np.linalg.norm(ctx.client_update[0] - g[0]))
    assert delta_norm <= 1.0 + 1e-5


def test_median_defense_rejects_outlier():
    from fltest.defenses.median import MedianDefense

    honest = [np.ones((5,), dtype=np.float32)]
    outlier = [np.full((5,), 999.0, dtype=np.float32)]
    uw = [(honest, 10), (honest, 10), (outlier, 10)]
    ctx = HookContext(cfg=_spec(), round=1, updates_and_weights=uw)
    MedianDefense().before_aggregate(ctx)
    agg, _ = ctx.updates_and_weights[0]
    assert np.allclose(agg[0], 1.0)  # median ignores the 999 outlier
