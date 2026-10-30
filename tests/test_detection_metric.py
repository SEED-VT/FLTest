"""Detection precision and recall against the ground truth a config already carries."""

from fltest.core.hook_context import HookContext
from fltest.core.config import TestConfig
from fltest.core.orchestrator import expand_run_specs
from fltest.metrics.detection import DetectionQualityListener


def _ctx(attacks, flagged, num_clients=8):
    spec = expand_run_specs(TestConfig(
        name="d", dataset="mnist", num_clients=num_clients, attacks=attacks,
        runs=[{"framework": "reference"}]))[0]
    ctx = HookContext(cfg=spec)
    if flagged is not None:
        ctx.history[5] = {"fldetector_detected_clients": flagged}
    return ctx


SIGN_FLIP_01 = [{"name": "sign_flip", "target_clients": [0, 1]}]


def test_perfect_detection_scores_one():
    ctx = _ctx(SIGN_FLIP_01, [0, 1])
    DetectionQualityListener().after_simulation(ctx)
    assert ctx.metrics["detection_precision"] == 1.0
    assert ctx.metrics["detection_recall"] == 1.0
    assert ctx.metrics["detection_false_positives"] == 0.0
    assert ctx.metrics["detection_missed"] == 0.0


def test_a_flagged_honest_client_is_a_false_positive():
    ctx = _ctx(SIGN_FLIP_01, [0, 1, 5])
    DetectionQualityListener().after_simulation(ctx)
    assert ctx.metrics["detection_precision"] == 2 / 3
    assert ctx.metrics["detection_recall"] == 1.0
    assert ctx.metrics["detection_false_positives"] == 1.0


def test_a_missed_attacker_lowers_recall():
    ctx = _ctx(SIGN_FLIP_01, [0])
    DetectionQualityListener().after_simulation(ctx)
    assert ctx.metrics["detection_precision"] == 1.0
    assert ctx.metrics["detection_recall"] == 0.5
    assert ctx.metrics["detection_missed"] == 1.0


def test_flagging_nobody_scores_zero_rather_than_a_vacuous_one():
    ctx = _ctx(SIGN_FLIP_01, [])
    DetectionQualityListener().after_simulation(ctx)
    assert ctx.metrics["detection_precision"] == 0.0
    assert ctx.metrics["detection_recall"] == 0.0


def test_records_nothing_without_a_detector():
    """An arm with no detector must leave the columns empty, not report a score of zero."""
    ctx = _ctx(SIGN_FLIP_01, None)
    DetectionQualityListener().after_simulation(ctx)
    assert ctx.metrics == {}


def test_a_server_side_privacy_attack_is_not_ground_truth():
    """dlg names the victim it inverts, so client 3 is not a client the detector owed us."""
    ctx = _ctx([{"name": "dlg", "target_clients": [3]}], [0, 1])
    DetectionQualityListener().after_simulation(ctx)
    assert ctx.metrics == {}


def test_an_attack_targeting_everyone_has_no_honest_client_to_score_against():
    ctx = _ctx([{"name": "sign_flip"}], [0, 1])
    DetectionQualityListener().after_simulation(ctx)
    assert ctx.metrics == {}


def test_little_is_enough_names_its_attackers_as_ground_truth():
    ctx = _ctx([{"name": "little_is_enough", "target_clients": [0, 1]}], [0, 1])
    DetectionQualityListener().after_simulation(ctx)
    assert ctx.metrics["detection_precision"] == 1.0
    assert ctx.metrics["detection_recall"] == 1.0


def test_fldetector_paper_metrics_count_every_client():
    """DACC, FPR and FNR as the FLDetector paper defines them, over all eight clients."""
    ctx = _ctx(SIGN_FLIP_01, [0, 5])
    DetectionQualityListener().after_simulation(ctx)
    assert ctx.metrics["detection_accuracy"] == 6 / 8  # client 1 missed, client 5 accused
    assert ctx.metrics["detection_fpr"] == 1 / 6
    assert ctx.metrics["detection_fnr"] == 1 / 2
