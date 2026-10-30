"""Detection quality: did a defense flag the clients the config actually attacked?

A detector's accuracy column measures the model, not the detection. A run whose accuracy
recovers after exclusion has *probably* flagged the right clients, but a detector that
flagged one honest client alongside the attackers would produce nearly the same curve.
This listener compares the flagged set against the ground truth the config already
carries, so the claim is measured rather than read off an accuracy trace.

Ground truth is the union of ``target_clients`` over the attacks that make a client
malicious. Server-side privacy attacks are excluded from that union: ``dlg`` and
``membership_inference`` name a *victim*, not an attacker, so counting them would mark
an honest client as one the detector should have caught.
"""

from __future__ import annotations

from typing import Optional, Set

from fltest.core.hook_context import HookContext
from fltest.core.registry import register_metric
from fltest.metrics.base import MetricListenerBaseClass

#: Attacks whose ``target_clients`` are the malicious clients themselves. An attack absent
#: from this set is server-side, and its targets are victims rather than adversaries.
MALICIOUS_CLIENT_ATTACKS = {
    "backdoor", "label_flip", "sign_flip", "gaussian", "model_replacement",
    "little_is_enough", "little_is_enough_backdoor", "little_is_enough_cropped_backdoor",
}

#: Metrics a detector records to announce what it flagged.
_DETECTED_KEYS = ("fldetector_detected_clients",)


def _ground_truth(spec) -> Optional[Set[int]]:
    """Malicious client IDs named by the config, or None when they cannot be determined."""
    attacks = [a for a in getattr(spec, "attacks", []) if a.name in MALICIOUS_CLIENT_ATTACKS]
    if not attacks:
        return None
    truth: Set[int] = set()
    for attack in attacks:
        if attack.target_clients is None:
            # The attack decides, and usually that means every client. There is no honest
            # client left to be a false positive, so precision and recall say nothing.
            return None
        truth.update(int(cid) for cid in attack.target_clients)
    return truth


@register_metric("detection")
class DetectionQualityListener(MetricListenerBaseClass):
    """Precision and recall of a defense's malicious-client detection.

    Records nothing unless the run has both a detector that announced a flagged set and a
    config naming which clients are malicious, so adding it to ``metrics`` is always safe.
    """

    HOOKS = ("after_simulation",)

    def after_simulation(self, ctx: HookContext) -> None:
        if ctx.cfg is None:
            return
        truth = _ground_truth(ctx.cfg)
        if truth is None:
            return

        flagged: Optional[Set[int]] = None
        detection_round = None
        for rnd in sorted(ctx.history):
            for key in _DETECTED_KEYS:
                value = ctx.history[rnd].get(key)
                if value is not None:
                    flagged = {int(cid) for cid in value}
                    detection_round = rnd
        if flagged is None:
            return

        hits = len(flagged & truth)
        # Zero-division convention matches scikit-learn's `zero_division=0`: a detector
        # that flagged nobody scores 0 rather than a vacuous 1.
        precision = hits / len(flagged) if flagged else 0.0
        recall = hits / len(truth) if truth else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        false_positives, missed = len(flagged - truth), len(truth - flagged)
        # The FLDetector paper's own three, over every client in the federation: an
        # unflagged client counts as classified benign.
        clients = max(int(ctx.cfg.num_clients), len(truth | flagged))
        honest = clients - len(truth)
        ctx.record(
            detection_precision=precision,
            detection_recall=recall,
            detection_f1=f1,
            detection_false_positives=float(false_positives),
            detection_missed=float(missed),
            detection_accuracy=(clients - false_positives - missed) / clients,
            detection_fpr=false_positives / honest if honest else 0.0,
            detection_fnr=missed / len(truth) if truth else 0.0,
        )
        ctx.extras["detection"] = {
            "malicious_clients": sorted(truth),
            "flagged_clients": sorted(flagged),
            "detection_round": detection_round,
        }
