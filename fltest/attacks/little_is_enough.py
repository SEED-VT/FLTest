"""A Little Is Enough: a Byzantine attack sized to the honest population's own spread.

Baruch, Baruch and Goldberg, *A Little Is Enough: Circumventing Defenses For Distributed
Learning* (NeurIPS 2019).

``gaussian`` and ``sign_flip`` perturb an update without reference to what the honest
updates look like, so any perturbation large enough to matter is also an outlier — which is
exactly the signal ``krum``, ``median`` and ``trimmed_mean`` key on. This attack measures
the benign population first and moves only as far as that population's own variance allows,
per coordinate:

``u' = mu - z * sigma``

With ``n`` participants and ``m`` attackers, ``s = floor(n/2 + 1) - m`` benign workers are
needed to complete a majority. A value ``z`` deviations out is covered by the
``(n - m) * (1 - Phi(z))`` honest workers lying further out still, so staying hidden needs
``(n - m) * (1 - Phi(z)) >= s``, i.e. ``Phi(z) <= (n - m - s) / (n - m)``; the default ``z``
is the largest value satisfying it. The crafted update therefore still falls inside the range
a majority of honest workers occupy: a distance-based rule keeps accepting it while the
aggregate drifts a little every round. Set ``z`` explicitly to sweep attack strength.

``sigma`` is the population standard deviation (NumPy's default). Reimplementations built on
``torch.std`` use the sample form, which is larger by ``sqrt(k / (k - 1))`` over ``k`` benign
clients, so the same ``z`` is a slightly wider step there than here.

Threat model: the paper's *full-knowledge* attacker, which sees the round's benign updates.
The attack runs at ``before_aggregate``, where FLTest exposes every submission, and rewrites
only the attackers' entries. Attacks attach before defenses, so a robust aggregator in the
same config still sees the crafted updates. NVFlare does not emit ``before_aggregate`` and
cannot run this attack.
"""

from __future__ import annotations

from statistics import NormalDist
from typing import List, Optional, Set

import numpy as np

from fltest.attacks.base import ThreatModelBaseClass
from fltest.core.hook_context import ClientSubmission, HookContext
from fltest.core.registry import register_attack


@register_attack("little_is_enough")
class LittleIsEnoughAttack(ThreatModelBaseClass):
    """Replace the attackers' updates with a perturbation of the benign mean."""

    HOOKS = ("before_aggregate", "after_aggregate")

    def __init__(self, z: Optional[float] = None, **params):
        super().__init__(**params)
        if z is not None and z < 0:
            raise ValueError("little_is_enough z must not be negative")
        if not self.target_clients:
            raise ValueError(
                "little_is_enough needs target_clients: it crafts the attackers' updates "
                "from the benign ones, so the adversarial clients must be named"
            )
        self.z = z
        self._asked = None  # benign statistics of the round, for scoring the aggregate

    def _attacker_positions(self, ctx: HookContext, count: int) -> Set[int]:
        """Positions in ``updates_and_weights`` held by the adversarial clients."""
        submissions = ctx.client_submissions
        if submissions is not None and len(submissions) == count:
            ids = [s.client_id for s in submissions]
            if all(cid is not None for cid in ids):
                return {pos for pos, cid in enumerate(ids) if self.targets(cid)}
        # Backends that do not identify submissions keep FLTest's positional convention.
        return {pos for pos in range(count) if self.targets(pos)}

    def _perturbation(self, num_participants: int, num_attackers: int) -> float:
        if self.z is not None:
            return float(self.z)
        s = num_participants // 2 + 1 - num_attackers
        benign = num_participants - num_attackers
        ratio = (benign - s) / benign
        # Below 0.5 no perturbation keeps the majority, and 1.0 would be unbounded.
        return NormalDist().inv_cdf(min(max(ratio, 0.5), 1.0 - 1e-9))

    @staticmethod
    def _layer_statistics(uw, benign: List[int], num_layers: int):
        """Per layer: a benign reference array, and the benign mean and deviation.

        Mean and deviation are None for a layer that is not floating point. Integer buffers
        (BatchNorm's num_batches_tracked and friends) carry no gradient signal, and
        perturbing one only truncates on the cast back.
        """
        stats = []
        for layer in range(num_layers):
            arrays = [np.asarray(uw[pos][0][layer]) for pos in benign]
            reference = arrays[0]
            if not np.issubdtype(reference.dtype, np.floating):
                stats.append((reference, None, None))
                continue
            stacked = np.stack([a.astype(np.float64) for a in arrays])
            stats.append((reference, stacked.mean(axis=0), stacked.std(axis=0)))
        return stats

    def _craft(self, ctx: HookContext, stats, z: float) -> List[np.ndarray]:
        """Algorithm 3: the benign mean moved ``z`` deviations, coordinate by coordinate."""
        return [
            reference.copy() if mean is None
            else (mean - z * deviation).astype(reference.dtype, copy=False)
            for reference, mean, deviation in stats
        ]

    def _craft_each(self, ctx: HookContext, stats, z: float, attackers) -> dict:
        """Each attacker's submission, by position. By default every attacker sends one craft."""
        crafted = self._craft(ctx, stats, z)
        return {pos: crafted for pos in attackers}

    def before_aggregate(self, ctx: HookContext) -> None:
        uw = ctx.updates_and_weights
        if not uw or len(uw) < 2:
            return
        attackers = self._attacker_positions(ctx, len(uw))
        benign = [pos for pos in range(len(uw)) if pos not in attackers]
        # A round may select no attacker, or leave no honest update to measure.
        if not attackers or not benign:
            return

        num_layers = len(uw[benign[0]][0])
        if any(len(uw[pos][0]) != num_layers for pos in range(len(uw))):
            raise ValueError("little_is_enough requires updates of matching length")

        z = self._perturbation(len(uw), len(attackers))
        stats = self._layer_statistics(uw, benign, num_layers)
        crafted = self._craft_each(ctx, stats, z, sorted(attackers))
        float_layers = [layer for layer, (_, mean, _) in enumerate(stats) if mean is not None]
        self._asked = (
            (float_layers,
             np.concatenate([stats[layer][1].ravel() for layer in float_layers]),
             np.concatenate([stats[layer][2].ravel() for layer in float_layers]), z)
            if float_layers else None
        )

        # The submission records describe what the server received, so they have to carry
        # the crafted arrays too — a detection defense reads them, and FLDetector rejects a
        # round whose records and aggregation inputs have drifted apart.
        records = list(ctx.client_submissions or ())
        aligned = len(records) == len(uw)
        for pos in attackers:
            submitted = list(crafted[pos])
            uw[pos] = (submitted, uw[pos][1])
            if aligned:
                records[pos] = ClientSubmission(
                    records[pos].client_id, tuple(submitted), records[pos].num_samples
                )
        if aligned:
            ctx.client_submissions = tuple(records)
        ctx.record(little_is_enough_z=z)

    def after_aggregate(self, ctx: HookContext) -> None:
        """Score the round: how far the aggregate actually moved, in benign sigmas.

        The attacker asked the aggregate to sit ``z`` standard deviations below the honest
        mean. Reading back where it landed says what the defense conceded — near ``z`` means
        the crafted update was absorbed, near 0 means it was filtered out.
        """
        asked, self._asked = self._asked, None
        if asked is None or ctx.new_global_state is None:
            return
        layers, mean, deviation, z = asked
        if max(layers) >= len(ctx.new_global_state):
            return
        aggregate = np.concatenate(
            [np.asarray(ctx.new_global_state[layer]).astype(np.float64).ravel()
             for layer in layers]
        )
        if aggregate.shape != mean.shape:
            return
        # Coordinates the honest clients all agree on carry no scale to measure against.
        movable = deviation > 0
        if not movable.any():
            return
        drift = float(np.mean((mean[movable] - aggregate[movable]) / deviation[movable]))
        ctx.record(little_is_enough_drift=drift)
        if z > 0:
            ctx.record(little_is_enough_absorption=drift / z)
