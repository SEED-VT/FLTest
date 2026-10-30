"""Unit tests for attacks/defenses operating on the canonical ndarray update."""

import numpy as np
import pytest
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


# --- little is enough ------------------------------------------------------------------

def _lie_round(values, attackers, num_clients=None, **kw):
    """Run the attack over one scalar-layer submission per client."""
    from fltest.attacks.little_is_enough import LittleIsEnoughAttack

    uw = [([np.full((2,), v, dtype=np.float32)], 10) for v in values]
    ctx = HookContext(
        cfg=_spec(num_clients=num_clients or len(values)), round=1, updates_and_weights=uw,
    )
    LittleIsEnoughAttack(target_clients=attackers, **kw).before_aggregate(ctx)
    return ctx


def test_little_is_enough_z_is_the_largest_step_that_stays_covered():
    """The property z is defined by: enough honest workers must lie further out than the craft.

    Checked against the criterion itself rather than against a remembered constant, so the
    test still means something if the derivation is ever revisited.
    """
    from statistics import NormalDist

    n, m = 50, 24
    s = n // 2 + 1 - m  # honest workers needed to complete a majority alongside the attackers
    z = _lie_round([1.0] * n, attackers=list(range(m))).metrics["little_is_enough_z"]

    def honest_further_out(step):
        return (n - m) * (1 - NormalDist().cdf(step))

    assert honest_further_out(z) >= s - 1e-9  # this step is still covered
    assert honest_further_out(z + 0.01) < s  # any larger step is not
    assert z == pytest.approx(1.426, abs=0.001)  # regression guard on the value itself


def test_little_is_enough_crafts_the_benign_mean_shifted_by_z_sigma():
    # Benign 1, 2, 3; the attackers' own 9.0 must not enter the statistics.
    ctx = _lie_round([1.0, 2.0, 3.0, 9.0, 9.0], attackers=[3, 4], z=1.5)

    expected = 2.0 - 1.5 * float(np.std([1.0, 2.0, 3.0]))
    for pos in (3, 4):
        update, num_samples = ctx.updates_and_weights[pos]
        assert np.allclose(update[0], expected)
        assert update[0].dtype == np.float32
        assert num_samples == 10  # sample weight survives, only the update is replaced
    for pos, benign in enumerate([1.0, 2.0, 3.0]):
        assert np.allclose(ctx.updates_and_weights[pos][0][0], benign)
    assert ctx.metrics["little_is_enough_z"] == 1.5


def test_little_is_enough_survives_krum_where_a_naive_outlier_does_not():
    """The point of the attack: stay inside the honest spread, so the filter keeps it."""
    from fltest.attacks.little_is_enough import LittleIsEnoughAttack
    from fltest.defenses.krum import KrumDefense

    rng = np.random.default_rng(0)
    num_clients, num_attackers, dim = 10, 4, 50
    honest = [
        [(np.ones(dim) + rng.normal(0, 0.1, dim)).astype(np.float32)]
        for _ in range(num_clients)
    ]
    attackers = list(range(num_clients - num_attackers, num_clients))

    crafted_round = HookContext(
        cfg=_spec(num_clients=num_clients), round=1,
        updates_and_weights=[(u, 1) for u in honest],
    )
    LittleIsEnoughAttack(target_clients=attackers).before_aggregate(crafted_round)
    crafted = crafted_round.updates_and_weights[attackers[0]][0][0]

    KrumDefense(num_byzantine=num_attackers).before_aggregate(crafted_round)
    assert np.allclose(crafted_round.updates_and_weights[0][0][0], crafted)

    # Same clients, same defense, but perturbed without regard for the honest spread.
    naive = [(list(u), 1) for u in honest]
    for pos in attackers:
        naive[pos] = ([(honest[pos][0] + rng.normal(0, 5.0, dim)).astype(np.float32)], 1)
    naive_round = HookContext(
        cfg=_spec(num_clients=num_clients), round=1, updates_and_weights=naive,
    )
    KrumDefense(num_byzantine=num_attackers).before_aggregate(naive_round)
    selected = naive_round.updates_and_weights[0][0][0]
    assert not any(np.allclose(selected, naive[pos][0][0]) for pos in attackers)


def test_little_is_enough_identifies_attackers_by_client_id():
    """Client selection makes position and identity diverge; identity must win."""
    from fltest.attacks.little_is_enough import LittleIsEnoughAttack
    from fltest.core.hook_context import ClientSubmission

    updates = [[np.full((2,), v, dtype=np.float32)] for v in (1.0, 2.0, 3.0)]
    selected = (5, 7, 9)  # round selected clients 5, 7 and 9, in that order
    ctx = HookContext(
        cfg=_spec(num_clients=10), round=1,
        updates_and_weights=[(u, 10) for u in updates],
        client_submissions=tuple(
            ClientSubmission(cid, tuple(u), 10) for cid, u in zip(selected, updates)
        ),
    )
    LittleIsEnoughAttack(target_clients=[7], z=0.0).before_aggregate(ctx)

    # Client 7 sits at position 1; with z=0 it submits the mean of clients 5 and 9.
    assert np.allclose(ctx.updates_and_weights[1][0][0], 2.0)
    assert np.allclose(ctx.updates_and_weights[0][0][0], 1.0)
    assert np.allclose(ctx.updates_and_weights[2][0][0], 3.0)


def _lie_scored_round(defense, z, num_clients=10, num_attackers=4, dim=400):
    """Craft, aggregate as the backend would, then let the attack score the outcome."""
    from fltest.attacks.little_is_enough import LittleIsEnoughAttack
    from fltest.data.utils import aggregate_ndarrays

    rng = np.random.default_rng(0)
    honest = [
        [(np.ones(dim) + rng.normal(0, 0.1, dim)).astype(np.float32)]
        for _ in range(num_clients)
    ]
    attackers = list(range(num_clients - num_attackers, num_clients))
    ctx = HookContext(
        cfg=_spec(num_clients=num_clients), round=1,
        updates_and_weights=[(list(u), 1) for u in honest],
    )
    attack = LittleIsEnoughAttack(target_clients=attackers, z=z)
    attack.before_aggregate(ctx)
    if defense is not None:
        defense.before_aggregate(ctx)
    ctx.new_global_state = aggregate_ndarrays(ctx.updates_and_weights)
    attack.after_aggregate(ctx)
    return ctx


def test_little_is_enough_absorption_under_plain_fedavg_is_the_attacker_fraction():
    """Averaging concedes exactly m/n of the requested shift, which anchors the metric."""
    ctx = _lie_scored_round(defense=None, z=1.5)
    assert ctx.metrics["little_is_enough_absorption"] == pytest.approx(0.4, abs=0.01)
    assert ctx.metrics["little_is_enough_drift"] == pytest.approx(0.6, abs=0.02)


def test_little_is_enough_absorption_reads_out_whether_krum_took_the_craft():
    from fltest.defenses.krum import KrumDefense

    taken = _lie_scored_round(defense=KrumDefense(num_byzantine=4), z=1.5)
    # Krum returns a single selected update, so absorption is all or nothing.
    assert taken.metrics["little_is_enough_absorption"] == pytest.approx(1.0, abs=0.01)

    filtered = _lie_scored_round(defense=KrumDefense(num_byzantine=4), z=3.0)
    assert filtered.metrics["little_is_enough_absorption"] == pytest.approx(0.0, abs=0.01)


def test_little_is_enough_scores_nothing_when_no_attacker_participates():
    from fltest.attacks.little_is_enough import LittleIsEnoughAttack

    ctx = _lie_round([1.0, 2.0, 3.0], attackers=[7], num_clients=10)
    ctx.new_global_state = [np.full((2,), 2.0, dtype=np.float32)]
    LittleIsEnoughAttack(target_clients=[7]).after_aggregate(ctx)
    assert "little_is_enough_drift" not in ctx.metrics


def test_little_is_enough_keeps_submission_records_aligned():
    """FLDetector reads the records and rejects a round that has drifted from them."""
    from fltest.attacks.little_is_enough import LittleIsEnoughAttack
    from fltest.core.hook_context import ClientSubmission

    updates = [[np.full((2,), v, dtype=np.float32)] for v in (1.0, 2.0, 3.0)]
    ctx = HookContext(
        cfg=_spec(num_clients=3), round=1,
        updates_and_weights=[(u, 10) for u in updates],
        client_submissions=tuple(
            ClientSubmission(cid, tuple(u), 10) for cid, u in enumerate(updates)
        ),
    )
    LittleIsEnoughAttack(target_clients=[2], z=1.0).before_aggregate(ctx)

    for record, (update, _) in zip(ctx.client_submissions, ctx.updates_and_weights):
        assert len(record.update) == len(update)
        assert all(a is b for a, b in zip(record.update, update))
    assert [r.client_id for r in ctx.client_submissions] == [0, 1, 2]
    assert not np.allclose(ctx.client_submissions[2].update[0], 3.0)  # record shows the craft


def test_little_is_enough_is_quiet_when_no_attacker_participates():
    ctx = _lie_round([1.0, 2.0, 3.0], attackers=[7], num_clients=10)
    for pos, benign in enumerate([1.0, 2.0, 3.0]):
        assert np.allclose(ctx.updates_and_weights[pos][0][0], benign)
    assert "little_is_enough_z" not in ctx.metrics


def test_little_is_enough_leaves_integer_buffers_intact():
    from fltest.attacks.little_is_enough import LittleIsEnoughAttack

    updates, weights = _updates_with_an_integer_buffer(num_clients=3)
    ctx = HookContext(cfg=_spec(num_clients=3), round=1,
                      updates_and_weights=list(zip(updates, weights)))
    LittleIsEnoughAttack(target_clients=[2], z=1.0).before_aggregate(ctx)

    crafted = ctx.updates_and_weights[2][0]
    assert crafted[1] == 7 and crafted[1].dtype == np.int64
    assert crafted[0].dtype == np.float32


def test_little_is_enough_requires_named_attackers():
    from fltest.attacks.little_is_enough import LittleIsEnoughAttack

    with pytest.raises(ValueError, match="target_clients"):
        LittleIsEnoughAttack()


# --- little is enough, backdoor variant ------------------------------------------------

def _lie_backdoor(z, alpha=1.0, sigma=0.01, attackers=(3, 4), num_clients=5, cropped=False,
                  **kw):
    """Honest MLP updates spread by ``sigma``, and an attack that has seen its own data."""
    from fltest.attacks.little_is_enough_backdoor import (
        LittleIsEnoughBackdoorAttack,
        LittleIsEnoughCroppedBackdoorAttack,
    )
    from fltest.data.models import get_model
    from fltest.data.utils import state_dict_to_ndarrays

    torch.manual_seed(0)
    base = state_dict_to_ndarrays(get_model("MLP", "", channels=1, deterministic=False).state_dict())
    rng = np.random.default_rng(0)
    uw = [([(w + rng.normal(0, sigma, w.shape)).astype(np.float32) for w in base], 10)
          for _ in range(num_clients)]
    loaders = {cid: [{"img": torch.randn(32, 1, 32, 32), "label": torch.randint(0, 10, (32,))}]
               for cid in range(num_clients)}
    spec = _spec(model_name="MLP", channels=1, client_lr=0.05, client_batch_size=32,
                 optimizer="SGD", seed=1)
    if cropped:
        attack = LittleIsEnoughCroppedBackdoorAttack(target_clients=list(attackers), z=z, **kw)
    else:
        attack = LittleIsEnoughBackdoorAttack(
            target_clients=list(attackers), z=z, alpha=alpha, samples=64, **kw)
    attack.on_data_distribute(HookContext(cfg=spec, dist_dict=loaders))
    ctx = HookContext(cfg=spec, round=1, updates_and_weights=[(list(u), n) for u, n in uw],
                      global_state=base)
    attack.before_aggregate(ctx)
    return ctx, uw, spec


def test_little_is_enough_backdoor_stays_inside_the_honest_spread():
    ctx, uw, _ = _lie_backdoor(z=0.5)
    honest = [uw[pos][0] for pos in (0, 1, 2)]
    crafted = ctx.updates_and_weights[3][0]
    moved = False
    for layer, value in enumerate(crafted):
        stacked = np.stack([h[layer].astype(np.float64) for h in honest])
        mean, bound = stacked.mean(axis=0), 0.5 * stacked.std(axis=0)
        assert np.all(value >= mean - bound - 1e-6) and np.all(value <= mean + bound + 1e-6)
        moved |= not np.allclose(value, mean, atol=1e-6)
        assert np.array_equal(value, ctx.updates_and_weights[4][0][layer])  # every attacker alike
    assert moved  # the clamp bounds the craft, it does not erase it
    for pos in (0, 1, 2):
        assert all(np.array_equal(a, b) for a, b in zip(ctx.updates_and_weights[pos][0], uw[pos][0]))


def test_little_is_enough_backdoor_crafts_a_working_trigger():
    """With the clamp opened wide, what was trained is what is sent: a model with the trigger."""
    from fltest.attacks.little_is_enough_backdoor import _stamp
    from fltest.data.models import get_model
    from fltest.data.utils import load_ndarrays_into

    ctx, _, _ = _lie_backdoor(z=1e6, target_label=3)
    model = get_model("MLP", "", channels=1, deterministic=False)
    load_ndarrays_into(model, ctx.updates_and_weights[3][0])
    with torch.no_grad():
        predictions = model(_stamp(torch.randn(64, 1, 32, 32), 5, 1.0)).argmax(dim=1)
    assert (predictions == 3).float().mean() > 0.9


def test_cropped_backdoor_gives_each_attacker_its_own_update_inside_the_spread():
    """The FLDetector paper's variant: per-attacker poisoned training, then a crop."""
    ctx, uw, _ = _lie_backdoor(z=0.5, cropped=True)
    honest = [uw[pos][0] for pos in (0, 1, 2)]
    first, second = ctx.updates_and_weights[3][0], ctx.updates_and_weights[4][0]
    assert not all(np.array_equal(a, b) for a, b in zip(first, second))
    for crafted in (first, second):
        for layer, value in enumerate(crafted):
            stacked = np.stack([h[layer].astype(np.float64) for h in honest])
            mean, bound = stacked.mean(axis=0), 0.5 * stacked.std(axis=0)
            assert np.all(value >= mean - bound - 1e-6) and np.all(value <= mean + bound + 1e-6)
    for pos in (0, 1, 2):
        assert all(np.array_equal(a, b) for a, b in zip(ctx.updates_and_weights[pos][0], uw[pos][0]))


def test_cropped_backdoor_trains_the_trigger_into_each_attackers_model():
    from fltest.attacks.little_is_enough_backdoor import _stamp
    from fltest.data.models import get_model
    from fltest.data.utils import load_ndarrays_into

    ctx, _, spec = _lie_backdoor(z=1e6, cropped=True, target_label=3)
    for pos in (3, 4):
        model = get_model("MLP", "", channels=1, deterministic=False)
        load_ndarrays_into(model, ctx.updates_and_weights[pos][0])
        with torch.no_grad():
            predictions = model(_stamp(torch.randn(64, 1, 32, 32), 5, 1.0)).argmax(dim=1)
        assert (predictions == 3).float().mean() > 0.9


def test_little_is_enough_backdoor_penalty_stays_finite_when_honest_clients_agree():
    """Equation 4 divides by z * sigma; a near-zero spread must not blow up the training."""
    ctx, _, _ = _lie_backdoor(z=0.2, alpha=0.2, sigma=1e-7)
    assert all(np.isfinite(layer).all() for layer in ctx.updates_and_weights[3][0])


def test_little_is_enough_backdoor_leaves_the_global_rng_alone():
    """Honest clients must shuffle as they would without the attacker in the run."""
    from fltest.attacks.little_is_enough_backdoor import LittleIsEnoughBackdoorAttack

    rng = np.random.default_rng(0)
    shapes = [(128, 1024), (128,), (10, 128), (10,)]  # the MLP's layers, built without torch
    uw = [([rng.normal(0, 0.1, s).astype(np.float32) for s in shapes], 10) for _ in range(4)]
    loaders = {3: [{"img": torch.zeros(8, 1, 32, 32), "label": torch.zeros(8, dtype=torch.long)}]}
    spec = _spec(model_name="MLP", channels=1, client_lr=0.05, client_batch_size=4,
                 optimizer="SGD", seed=1)

    torch.manual_seed(5)
    expected = torch.rand(3)
    torch.manual_seed(5)
    attack = LittleIsEnoughBackdoorAttack(target_clients=[3], z=0.5, samples=8, epochs=1)
    attack.on_data_distribute(HookContext(cfg=spec, dist_dict=loaders))
    attack.before_aggregate(HookContext(cfg=spec, round=1, updates_and_weights=uw))
    assert torch.equal(torch.rand(3), expected)


def test_little_is_enough_backdoor_scores_attack_success_rate():
    from fltest.attacks.little_is_enough_backdoor import LittleIsEnoughBackdoorAttack

    class AlwaysTarget(torch.nn.Module):
        def forward(self, x):
            return torch.nn.functional.one_hot(torch.full((len(x),), 2), 10).float()

    test_data = [{"img": torch.zeros(4, 1, 32, 32), "label": torch.tensor([2, 0, 1, 3])}]
    ctx = HookContext(cfg=_spec(), round=1, global_state=[np.zeros(1)], model=AlwaysTarget(),
                      test_data=test_data)
    LittleIsEnoughBackdoorAttack(target_clients=[0], target_label=2).after_round(ctx)
    assert ctx.metrics["attack_success_rate"] == 1.0  # the sample already labelled 2 is excluded


def test_little_is_enough_backdoor_needs_images():
    from fltest.attacks.little_is_enough_backdoor import LittleIsEnoughBackdoorAttack

    text = {0: [{"input_ids": torch.zeros(2, 4, dtype=torch.long), "label": torch.zeros(2)}]}
    with pytest.raises(ValueError, match="image dataset"):
        LittleIsEnoughBackdoorAttack(target_clients=[0]).on_data_distribute(
            HookContext(cfg=_spec(), dist_dict=text))


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
