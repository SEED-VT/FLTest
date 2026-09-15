"""Fixed-point secure aggregation over a finite ring — the variant that can go wrong.

``secure_aggregation`` masks in real arithmetic, so the only thing it can teach is whether
blinding the uploaded update blunts an attack. Deployed secure aggregation does not work in
real arithmetic: shares live in ``Z_modulus``, floats are encoded as fixed point, and the
protocol therefore carries three failure modes that a float simulation makes invisible.

1. **Quantization error.** ``quant_bits`` fractional bits round every coordinate. Too few
   bits and the aggregate drifts from plain FedAvg by more than the update itself.
2. **Overflow.** Encoded values wrap around ``modulus`` instead of saturating. A wrap is
   silent — the aggregate stays finite and plausible while being wildly wrong — and it is
   reproduced here rather than clamped away, because clamping is what hides the bug.
3. **Masks that fail to cancel.** Pairwise masks telescope only across exactly the set that
   produced them. ``dropout_rate`` removes clients *after* they masked, leaving their
   partners' halves in the sum.

The whole protocol is simulated at ``before_aggregate``, where the server holds every
client's update. That placement is deliberate: it is the only point both backends reach in
a single process, and it is what lets the defense compute plain FedAvg alongside the
protocol's output and record the exact gap. ``mpc_agg_max_abs_error`` is that gap — a
strict numeric oracle, not an accuracy threshold. A correctly parameterised run drives it
to the quantization floor; a broken one does not.

The cost of that placement is that the client-side view is *not* simulated here: an attack
reading ``ctx.updates_and_weights`` still sees plaintext updates, because attacks attach
before defenses on the same hook. Use ``secure_aggregation`` for adversary-view
experiments and this one for arithmetic correctness.

**Limits.** Same as ``secure_aggregation``: masks derive from a seed this process knows,
there is no key agreement and no cryptography. ``dropout_rate`` demonstrates the failure
mode that threshold secret sharing exists to repair — it does **not** implement the repair,
and a run with dropouts is a run whose aggregate is knowingly wrong.
"""

from __future__ import annotations

import math
from typing import List

import numpy as np

from fltest.core.hook_context import HookContext
from fltest.core.registry import register_defense
from fltest.defenses._robust import replace_with
from fltest.defenses._secagg import (
    MAX_MODULUS,
    is_maskable,
    decode_fixed_point,
    encode_fixed_point,
    max_abs,
    overflow_fraction,
    pairwise_ring_mask,
)
from fltest.defenses.base import PPFLBaseClass

DEFAULT_MODULUS = 1 << 32


@register_defense("mpc_aggregation")
class MPCAggregationDefense(PPFLBaseClass):
    """Fixed-point masked aggregation in ``Z_modulus``, with the error against FedAvg recorded."""

    HOOKS = ("before_aggregate",)

    def __init__(
        self,
        quant_bits: int = 16,
        modulus: int = DEFAULT_MODULUS,
        dropout_rate: float = 0.0,
        seed: int = 0,
        mask: bool = True,
        **params,
    ):
        super().__init__(**params)
        if quant_bits < 0:
            raise ValueError("quant_bits must be non-negative")
        if modulus < 2 or modulus > MAX_MODULUS:
            raise ValueError(f"modulus must be in [2, {MAX_MODULUS}]; got {modulus}")
        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError("dropout_rate must be in [0, 1)")
        self.quant_bits = int(quant_bits)
        self.modulus = int(modulus)
        self.dropout_rate = float(dropout_rate)
        self.seed = int(seed)
        self.mask = bool(mask)

    # ---- protocol simulation ----

    def before_aggregate(self, ctx: HookContext) -> None:
        uw = ctx.updates_and_weights
        if not uw or ctx.round is None:
            return

        updates = [u for u, _ in uw]
        weights = [max(int(n), 1) for _, n in uw]
        num_clients = len(uw)
        template = updates[0]
        participants = list(range(num_clients))

        # Clients that mask and then vanish. Chosen deterministically from (seed, round) so
        # the run stays reproducible; never all of them, or there is nothing to aggregate.
        num_dropped = min(int(round(self.dropout_rate * num_clients)), num_clients - 1)
        dropped: List[int] = []
        if num_dropped > 0:
            rng = np.random.default_rng([self.seed, int(ctx.round), 0xD0])
            dropped = sorted(int(i) for i in rng.choice(num_clients, num_dropped, replace=False))
        survivors = [i for i in participants if i not in dropped]

        # Plain FedAvg over the *survivors*. Restricting the reference to the same clients
        # the protocol actually sums is what makes the recorded error mean "the protocol's
        # arithmetic went wrong" rather than "some clients' data is missing".
        surviving_total = float(sum(weights[i] for i in survivors)) or 1.0
        reference = [
            sum(updates[i][layer].astype(np.float64) * weights[i] for i in survivors)
            / surviving_total
            for layer in range(len(template))
        ]

        # Overflow is a property of the *sum*, not of the individual shares. A share that
        # wraps is harmless as long as sum_i n_i * x_i lands back inside the ring's signed
        # band — that is just how two's-complement modular arithmetic works, and flagging
        # per-client wraps would cry wolf on runs that are exactly right. What cannot be
        # recovered is the summed value leaving the band, so that is what is measured.
        weighted_sum = [
            ref * surviving_total
            for ref, tmpl in zip(reference, template)
            if is_maskable(tmpl)
        ]
        overflow = max(
            (overflow_fraction(w, self.quant_bits, self.modulus) for w in weighted_sum),
            default=0.0,
        )

        # 1. Every participant — dropouts included — encodes n_i * x_i and blinds it. The
        #    weighting happens before encoding because the ring has no fractional weights:
        #    the server can only sum and divide by the public total afterwards.
        max_scaled = 0.0
        shares: List[List[np.ndarray]] = []
        for cid in participants:
            weighted = [
                u.astype(np.float64) * weights[cid] if is_maskable(tmpl) else np.zeros_like(u, np.float64)
                for u, tmpl in zip(updates[cid], template)
            ]
            max_scaled = max(max_scaled, max_abs(weighted) * float(1 << self.quant_bits))
            share = [encode_fixed_point(w, self.quant_bits, self.modulus) for w in weighted]
            if self.mask and num_clients >= 2:
                mask = pairwise_ring_mask(
                    template, cid, participants, self.seed, ctx.round, self.modulus
                )
                share = [(s + m) % self.modulus for s, m in zip(share, mask)]
            shares.append(share)

        # 2. The server sums only what it received. Reducing after each client keeps the
        #    running value inside int64 for any accepted modulus.
        acc = [np.zeros(a.shape, dtype=np.int64) for a in template]
        for cid in survivors:
            acc = [(a + s) % self.modulus for a, s in zip(acc, shares[cid])]

        # 3. Decode and divide by the surviving sample total. Integer entries never entered
        #    the ring (see `is_maskable`), so they take the plaintext average the backend
        #    would have produced — they are counters, not learned parameters.
        aggregate = [
            (
                decode_fixed_point(a, self.quant_bits, self.modulus) / surviving_total
                if is_maskable(template[layer])
                else reference[layer]
            ).astype(template[layer].dtype)
            for layer, a in enumerate(acc)
        ]

        error = max_abs([
            agg.astype(np.float64) - ref
            for agg, ref, tmpl in zip(aggregate, reference, template)
            if is_maskable(tmpl)
        ])
        scale = max_abs(reference)
        ctx.record(
            mpc_agg_max_abs_error=error,
            mpc_agg_rel_error=(error / scale) if scale else 0.0,
            mpc_overflow_rate=overflow,
            mpc_dropouts=float(len(dropped)),
            # log2 of the largest per-client encoded magnitude. The float64 encode is exact
            # below 53 bits and starts dropping low-order bits above it — a quieter failure
            # than overflow, and one no other metric here would show.
            mpc_max_encoded_bits=(math.log2(max_scaled) if max_scaled > 0 else 0.0),
        )
        ctx.extras["mpc_aggregation"] = {
            "quant_bits": self.quant_bits,
            "modulus": self.modulus,
            "dropped_clients": dropped,
            "max_abs_error": error,
            "overflow_rate": overflow,
        }

        replace_with(ctx, aggregate)
