"""Pairwise-masked secure aggregation (Bonawitz et al., CCS 2017), float / lossless.

Each client adds a mask that cancels once the server averages: the server sees a uniformly
blinded update per client and only the aggregate is meaningful. This is the *baseline*
variant — real arithmetic, no quantization — so it is what belongs in a privacy comparison
where the question is "does masking blunt gradient inversion" rather than "does the
protocol's arithmetic hold up". For the arithmetic questions use ``mpc_aggregation``.

**Where the mask lands.** FedAvg's aggregate is ``sum_i (n_i / N) * x_i``. For the masks to
cancel under that weighting, client ``i`` uploads ``x_i + m_i / n_i``, which makes the
aggregate ``FedAvg(x) + (1 / N) * sum_i m_i`` and the second term zero. This is the same
protocol as the textbook formulation where clients upload ``n_i * x_i + m_i`` and the
server divides by ``N``; FLTest's wire format carries the unweighted update, so the
equivalent mask is divided by ``n_i`` instead.

One consequence is worth stating plainly, because it decides whether a configuration
actually hides anything: the on-the-wire mask has standard deviation
``mask_scale * sqrt(P - 1) / n_i``. A client holding a lot of data gets a proportionally
smaller mask. ``mask_scale`` therefore has to be set relative to the client shard size, not
to the parameter scale. The defense records ``secagg_mask_to_update_ratio`` every round so
this is measured rather than assumed — a ratio far below 1 means the mask is cosmetic.

**What this does and does not show.**

* It *does* reproduce the server's view: at ``before_aggregate`` — where the ``dlg`` attack
  with ``source: shared_update`` reads the uploaded update — the value is masked, so
  reconstruction degrades. That is a faithful simulation of the honest-but-curious server.
* It does *not* demonstrate cryptographic security. The masks come from a seed this
  process knows, there is no key agreement, no threshold secret sharing, and no dropout
  recovery. Claims of the form "SecAgg protects X" cannot be supported by this simulation;
  claims of the form "under the simulated adversary view, attack Y no longer succeeds" can.

**Participation.** Masks cancel only across exactly the participant set they were built
for. FLTest's backends use full participation (the reference loop iterates every client;
the Flower strategy sets ``fraction_fit=1.0``), so the set is ``range(num_clients)``. If a
run ever aggregates a different number of updates, the leftover masks do *not* cancel and
the aggregate is silently wrong — so ``before_aggregate`` checks the count and records
``secagg_participant_mismatch`` rather than letting it pass unnoticed.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from fltest.core.hook_context import HookContext
from fltest.core.registry import register_defense
from fltest.defenses._secagg import is_maskable, l2, max_abs, pairwise_float_mask
from fltest.defenses.base import PPFLBaseClass


@register_defense("secure_aggregation")
class SecureAggregationDefense(PPFLBaseClass):
    """Float pairwise masking; cancels under FedAvg's sample-weighted mean."""

    HOOKS = ("after_client_train", "before_aggregate")

    def __init__(
        self,
        mask_scale: float = 1.0,
        seed: int = 0,
        num_participants: Optional[int] = None,
        **params,
    ):
        super().__init__(**params)
        self.mask_scale = float(mask_scale)
        self.seed = int(seed)
        # None => take the participant set from the run's client count at hook time.
        self.num_participants = num_participants

    # ---- helpers ----

    def _participants(self, ctx: HookContext) -> List[int]:
        if self.num_participants is not None:
            return list(range(int(self.num_participants)))
        if ctx.cfg is None:
            return []
        return list(range(int(ctx.cfg.num_clients)))

    # ---- client side: blind the update before it is uploaded ----

    def after_client_train(self, ctx: HookContext) -> None:
        if ctx.client_update is None or ctx.client_id is None or ctx.round is None:
            return
        participants = self._participants(ctx)
        if ctx.client_id not in participants:
            # Outside the masked set: masking here would leave a residue nothing cancels.
            # Distinct from the server-side count check, which runs later and would
            # otherwise overwrite this on the same key.
            ctx.record(secagg_client_outside_mask_set=1.0)
            return
        if len(participants) < 2 or self.mask_scale <= 0.0:
            return

        update = ctx.client_update
        # n_i scales the mask so the sample-weighted mean cancels it exactly (see module doc).
        n = max(int(ctx.num_samples or 1), 1)
        mask = pairwise_float_mask(
            update, ctx.client_id, participants, self.seed, ctx.round, self.mask_scale
        )
        # Integer entries stay untouched: a float mask cast back to int64 truncates, so the
        # halves no longer cancel and the aggregate silently drifts. See `is_maskable`.
        wire_mask = [
            (m / n) if is_maskable(u) else np.zeros_like(m) for u, m in zip(update, mask)
        ]
        ctx.client_update = [
            (u.astype(np.float64) + m).astype(u.dtype) if is_maskable(u) else u
            for u, m in zip(update, wire_mask)
        ]

        update_norm = l2(update)
        ctx.record(
            secagg_mask_to_update_ratio=(l2(wire_mask) / update_norm) if update_norm else 0.0,
        )

    # ---- server side: verify, do not mutate (FedAvg itself removes the masks) ----

    def before_aggregate(self, ctx: HookContext) -> None:
        uw = ctx.updates_and_weights
        if not uw or ctx.round is None:
            return
        participants = self._participants(ctx)
        mismatch = float(len(uw) != len(participants))

        # The masks are a deterministic function of (seed, round, participants), so the
        # server can re-derive the whole set and check the property the protocol rests on:
        # the weighted combination of every client's mask must be zero. This catches a
        # broken sign convention or participant set without ever seeing a plaintext update.
        residual = 0.0
        if participants and len(participants) >= 2 and self.mask_scale > 0.0:
            template = uw[0][0]
            total = float(sum(n for _, n in uw)) or 1.0
            acc = [np.zeros(a.shape, dtype=np.float64) for a in template]
            for cid in participants:
                mask = pairwise_float_mask(
                    template, cid, participants, self.seed, ctx.round, self.mask_scale
                )
                for i, m in enumerate(mask):
                    if is_maskable(template[i]):
                        acc[i] += m / total
            residual = max_abs(acc)

        ctx.record(
            secagg_participants=float(len(uw)),
            secagg_participant_mismatch=mismatch,
            secagg_mask_residual=residual,
        )
        ctx.extras["secure_aggregation"] = {
            "participants_expected": len(participants),
            "participants_observed": len(uw),
            "mask_residual": residual,
            "mask_scale": self.mask_scale,
        }
