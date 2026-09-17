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


# --- secure aggregation ----------------------------------------------------------------

def _fedavg(updates_and_weights):
    """Same sample-weighted mean the backends apply after ``before_aggregate``."""
    total = float(sum(n for _, n in updates_and_weights))
    layers = len(updates_and_weights[0][0])
    out = [np.zeros_like(updates_and_weights[0][0][i], dtype=np.float64) for i in range(layers)]
    for params, n in updates_and_weights:
        for i in range(layers):
            out[i] += params[i].astype(np.float64) * (n / total)
    return out


def _fake_updates(num_clients=4, seed=0):
    rng = np.random.default_rng(seed)
    updates = [
        [rng.normal(0, 0.05, (40,)).astype(np.float32), rng.normal(0, 0.05, (6,)).astype(np.float32)]
        for _ in range(num_clients)
    ]
    weights = [120, 340, 55, 900][:num_clients]  # deliberately unequal shard sizes
    return updates, weights


def _mask_all(defense, updates, weights, rnd=1):
    """Run the client-side hook for every client, returning the uploaded updates."""
    masked = []
    for cid, (update, n) in enumerate(zip(updates, weights)):
        ctx = HookContext(cfg=_spec(num_clients=len(updates)), round=rnd, client_id=cid,
                          client_update=[a.copy() for a in update], num_samples=n)
        defense.after_client_train(ctx)
        masked.append(ctx.client_update)
    return masked


def test_secure_aggregation_masks_cancel_under_weighted_fedavg():
    from fltest.defenses.secure_aggregation import SecureAggregationDefense

    updates, weights = _fake_updates()
    plain = _fedavg(list(zip(updates, weights)))
    masked = _mask_all(SecureAggregationDefense(mask_scale=50.0, seed=7), updates, weights)
    secure = _fedavg(list(zip(masked, weights)))

    # Cancellation is exact in real arithmetic; the wire format is float32, so what is left
    # is rounding. Assert it is negligible against the aggregate's own scale.
    scale = max(float(np.max(np.abs(p))) for p in plain)
    error = max(float(np.max(np.abs(a - b))) for a, b in zip(secure, plain))
    assert error < 1e-4 * scale


def test_secure_aggregation_actually_blinds_the_upload():
    """A defense whose masks cancel but that hides nothing would pass the test above."""
    from fltest.defenses.secure_aggregation import SecureAggregationDefense

    updates, weights = _fake_updates()
    masked = _mask_all(SecureAggregationDefense(mask_scale=5000.0, seed=7), updates, weights)
    for original, uploaded in zip(updates, masked):
        assert not np.allclose(original[0], uploaded[0])


def test_secure_aggregation_flags_participant_mismatch():
    """Masks cancel only across the set they were built for, so a short round must be loud."""
    from fltest.defenses.secure_aggregation import SecureAggregationDefense

    updates, weights = _fake_updates()
    defense = SecureAggregationDefense(mask_scale=50.0, seed=7)

    ctx = HookContext(cfg=_spec(num_clients=4), round=1, updates_and_weights=list(zip(updates, weights)))
    defense.before_aggregate(ctx)
    assert ctx.metrics["secagg_participant_mismatch"] == 0.0
    assert ctx.metrics["secagg_mask_residual"] < 1e-12  # sign convention is sound

    ctx = HookContext(cfg=_spec(num_clients=4), round=1,
                      updates_and_weights=list(zip(updates[:3], weights[:3])))
    defense.before_aggregate(ctx)
    assert ctx.metrics["secagg_participant_mismatch"] == 1.0


def test_mpc_aggregation_matches_fedavg_at_the_quantization_floor():
    from fltest.defenses.mpc_aggregation import MPCAggregationDefense

    updates, weights = _fake_updates()
    plain = _fedavg(list(zip(updates, weights)))
    ctx = HookContext(cfg=_spec(num_clients=4), round=1, updates_and_weights=list(zip(updates, weights)))
    MPCAggregationDefense(quant_bits=16, modulus=1 << 32, seed=7).before_aggregate(ctx)

    aggregate, _ = ctx.updates_and_weights[0]
    error = max(float(np.max(np.abs(a.astype(np.float64) - b))) for a, b in zip(aggregate, plain))
    assert error < 1e-4
    assert ctx.metrics["mpc_agg_max_abs_error"] == error
    assert ctx.metrics["mpc_overflow_rate"] == 0.0


def test_mpc_aggregation_is_independent_of_the_mask_seed():
    """The property ``secagg_lossless`` asserts: in the ring, cancellation is exact."""
    from fltest.defenses.mpc_aggregation import MPCAggregationDefense

    updates, weights = _fake_updates()
    results = []
    for seed in (1, 2, 3):
        ctx = HookContext(cfg=_spec(num_clients=4), round=1,
                          updates_and_weights=list(zip(updates, weights)))
        MPCAggregationDefense(quant_bits=16, modulus=1 << 32, seed=seed).before_aggregate(ctx)
        results.append(ctx.updates_and_weights[0][0])
    for other in results[1:]:
        for a, b in zip(results[0], other):
            assert np.array_equal(a, b)  # bit-identical, not merely close


def test_mpc_aggregation_reports_lower_precision_as_larger_error():
    from fltest.defenses.mpc_aggregation import MPCAggregationDefense

    updates, weights = _fake_updates()
    errors = {}
    for quant_bits in (4, 10, 16):
        ctx = HookContext(cfg=_spec(num_clients=4), round=1,
                          updates_and_weights=list(zip(updates, weights)))
        MPCAggregationDefense(quant_bits=quant_bits, modulus=1 << 40, seed=7).before_aggregate(ctx)
        errors[quant_bits] = ctx.metrics["mpc_agg_max_abs_error"]
    assert errors[4] > errors[10] > errors[16]


def test_mpc_aggregation_detects_ring_overflow():
    """A ring too small to hold the summed aggregate wraps silently — the metric must not."""
    from fltest.defenses.mpc_aggregation import MPCAggregationDefense

    updates, weights = _fake_updates()
    plain = _fedavg(list(zip(updates, weights)))
    ctx = HookContext(cfg=_spec(num_clients=4), round=1, updates_and_weights=list(zip(updates, weights)))
    MPCAggregationDefense(quant_bits=16, modulus=1 << 20, seed=7).before_aggregate(ctx)

    assert ctx.metrics["mpc_overflow_rate"] > 0.0
    aggregate, _ = ctx.updates_and_weights[0]
    error = max(float(np.max(np.abs(a.astype(np.float64) - b))) for a, b in zip(aggregate, plain))
    assert error > 1e-3  # wraparound, not a rounding difference


def test_mpc_aggregation_dropouts_leave_masks_uncancelled():
    """Dropout recovery is deliberately not implemented; the damage must be visible."""
    from fltest.defenses.mpc_aggregation import MPCAggregationDefense

    updates, weights = _fake_updates()
    ctx = HookContext(cfg=_spec(num_clients=4), round=1, updates_and_weights=list(zip(updates, weights)))
    MPCAggregationDefense(quant_bits=16, modulus=1 << 40, dropout_rate=0.3, seed=7).before_aggregate(ctx)

    assert ctx.metrics["mpc_dropouts"] == 1.0
    assert ctx.metrics["mpc_agg_max_abs_error"] > 1.0  # orders of magnitude past the update scale


def test_fixed_point_round_trip_across_the_accepted_modulus_range():
    """Encoding and decoding must be exact everywhere the constructor accepts a modulus.

    Both halves of this once used float64 arithmetic, which drops low-order bits above
    2**53 — so a *larger* ring was quietly less accurate, which is the opposite of what a
    user tuning `modulus` would expect.
    """
    from fltest.defenses._secagg import MAX_MODULUS, decode_fixed_point, encode_fixed_point

    quant_bits = 16
    for modulus in (1 << 32, 1 << 52, 1 << 56, MAX_MODULUS):
        values = np.array([
            -0.1883544921875,                        # not aligned to the float64 ULP up there
            0.5,
            -(modulus // 2) / (1 << quant_bits),      # the most-negative representable value
        ])
        back = decode_fixed_point(encode_fixed_point(values, quant_bits, modulus),
                                  quant_bits, modulus)
        assert np.array_equal(back, values), f"round trip lost precision at modulus={modulus}"


def test_mpc_aggregation_accuracy_does_not_degrade_with_a_larger_ring():
    from fltest.defenses.mpc_aggregation import MPCAggregationDefense

    updates, weights = _fake_updates()
    errors = []
    for modulus in (1 << 32, 1 << 56, 1 << 60):
        ctx = HookContext(cfg=_spec(num_clients=4), round=1,
                          updates_and_weights=list(zip(updates, weights)))
        MPCAggregationDefense(quant_bits=16, modulus=modulus, seed=7).before_aggregate(ctx)
        errors.append(ctx.metrics["mpc_agg_max_abs_error"])
    assert max(errors) == min(errors)  # all at the quantization floor, none worse


def _updates_with_an_integer_buffer(num_clients=3):
    """BatchNorm contributes an int64 ``num_batches_tracked``; HF models carry similar buffers."""
    return [
        [np.array([0.1, 0.2], dtype=np.float32), np.array(7, dtype=np.int64)]
        for _ in range(num_clients)
    ], [100, 200, 300][:num_clients]


def test_secure_aggregation_leaves_integer_buffers_intact():
    """A float mask cast back to int64 truncates, so the halves stop cancelling."""
    from fltest.defenses.secure_aggregation import SecureAggregationDefense

    updates, weights = _updates_with_an_integer_buffer()
    masked = _mask_all(SecureAggregationDefense(mask_scale=500.0, seed=1), updates, weights)

    aggregated = _fedavg(list(zip(masked, weights)))
    assert aggregated[1] == 7.0                                   # counter survives untouched
    assert not np.allclose(masked[0][0], updates[0][0])           # float entry still masked


def test_mpc_aggregation_leaves_integer_buffers_intact():
    from fltest.defenses.mpc_aggregation import MPCAggregationDefense

    updates, weights = _updates_with_an_integer_buffer()
    ctx = HookContext(cfg=_spec(num_clients=3), round=1, updates_and_weights=list(zip(updates, weights)))
    MPCAggregationDefense(quant_bits=16, modulus=1 << 32, seed=1).before_aggregate(ctx)

    aggregate, _ = ctx.updates_and_weights[0]
    assert aggregate[1] == 7 and aggregate[1].dtype == np.int64
    assert np.allclose(aggregate[0], [0.1, 0.2], atol=1e-4)
