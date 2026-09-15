"""Shared machinery for the two secure-aggregation defenses.

Both defenses build **pairwise masks** in the style of Bonawitz et al. (CCS 2017): every
unordered pair of participants ``(u, v)`` derives one shared pseudo-random vector, the
lower-indexed party adds it and the higher-indexed party subtracts it, so the masks
telescope to zero once every participant's contribution is summed.

The one design constraint that matters here is that a mask must be reproducible **without
shared state**. FLTest's Flower backend runs client hooks inside Ray workers and rebuilds
every plugin from the picklable ``RunSpec``, so two clients can never hand each other a
key. Instead each pair mask is a pure function of ``(base_seed, round, lo, hi, layer)`` —
both parties derive the identical vector independently, and the server can re-derive the
whole set for verification. That is a *simulation* of the key-agreement step, not an
implementation of it; see the module docstrings of the two defenses for what that does
and does not demonstrate.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

#: Largest modulus we accept. Partial sums are reduced after every client, so the running
#: value stays below ``2 * modulus``; keeping that inside int64 bounds the ring here.
MAX_MODULUS = 1 << 62


def _pair_rng(base_seed: int, rnd: int, lo: int, hi: int, layer: int) -> np.random.Generator:
    """The generator both members of pair ``(lo, hi)`` derive for one layer of one round."""
    return np.random.default_rng([int(base_seed), int(rnd), int(lo), int(hi), int(layer)])


def pairwise_float_mask(
    template: Sequence[np.ndarray],
    cid: int,
    participants: Sequence[int],
    base_seed: int,
    rnd: int,
    scale: float,
) -> List[np.ndarray]:
    """Client ``cid``'s float mask: ``sum_v sign(cid, v) * g_{cid,v}``.

    ``sign`` is ``+1`` when ``cid`` is the lower index of the pair and ``-1`` otherwise, so
    summing this over all participants gives exactly zero in real arithmetic.

    Cost is O(len(participants) x len(template)) per client, i.e. O(P^2) per round overall.
    That is fine for the client counts FLTest simulates and is the price of deriving masks
    with no inter-client channel.
    """
    mask = [np.zeros(a.shape, dtype=np.float64) for a in template]
    for other in participants:
        if other == cid:
            continue
        lo, hi = (cid, other) if cid < other else (other, cid)
        sign = 1.0 if cid < other else -1.0
        for layer, arr in enumerate(template):
            rng = _pair_rng(base_seed, rnd, lo, hi, layer)
            mask[layer] += sign * rng.normal(0.0, scale, size=arr.shape)
    return mask


def pairwise_ring_mask(
    template: Sequence[np.ndarray],
    cid: int,
    participants: Sequence[int],
    base_seed: int,
    rnd: int,
    modulus: int,
) -> List[np.ndarray]:
    """Client ``cid``'s mask in ``Z_modulus`` — the integer analogue of the float mask.

    Cancellation here is *exact*, not approximate: the pair vector is drawn once as an
    integer and added by one party and subtracted by the other, so the ring sum is zero
    with no rounding to argue about. This is why the fixed-point variant is the one that
    supports a strict-equality oracle.
    """
    mask = [np.zeros(a.shape, dtype=np.int64) for a in template]
    for other in participants:
        if other == cid:
            continue
        lo, hi = (cid, other) if cid < other else (other, cid)
        sign = 1 if cid < other else -1
        for layer, arr in enumerate(template):
            rng = _pair_rng(base_seed, rnd, lo, hi, layer)
            draw = rng.integers(0, modulus, size=arr.shape, dtype=np.int64)
            mask[layer] = (mask[layer] + sign * draw) % modulus
    return mask


def encode_fixed_point(x: np.ndarray, quant_bits: int, modulus: int) -> np.ndarray:
    """Map floats to ``Z_modulus`` as two's-complement fixed point with ``quant_bits`` fractional bits.

    Values outside the representable band ``[-modulus / 2^(quant_bits+1), +...)`` wrap
    around instead of saturating — that wraparound *is* the overflow failure mode real
    fixed-point MPC exhibits, so it is reproduced rather than clamped away.
    """
    scaled = np.rint(np.asarray(x, dtype=np.float64) * float(1 << quant_bits))
    if scaled.size and np.max(np.abs(scaled)) >= 2.0 ** 63:
        raise ValueError(
            "fixed-point encoding overflowed int64: |value| * 2**quant_bits reached "
            f"{np.max(np.abs(scaled)):.3g}. Lower quant_bits, or check the updates are finite."
        )
    # The modulo runs in int64, not float64: above 2**53 a float64 remainder drops low-order
    # bits, which made a *larger* modulus quietly less accurate instead of more.
    return np.mod(scaled.astype(np.int64), modulus)


def decode_fixed_point(q: np.ndarray, quant_bits: int, modulus: int) -> np.ndarray:
    """Inverse of :func:`encode_fixed_point`, lifting the ring element to a signed value."""
    # Two details that both cost precision if got wrong:
    #   `>=`, not `>` — modulus/2 is the most-negative value the ring represents, and encode()
    #   maps -modulus/2 onto it, so treating it as positive breaks the round trip there.
    #   Centering in int64 before the float cast — a ring element near a large modulus does
    #   not survive float64, so subtracting the modulus afterwards subtracts from a number
    #   that has already lost its low-order bits.
    q = np.asarray(q, dtype=np.int64)
    half = modulus // 2
    centered = np.where(q >= half, q - modulus, q)
    return centered.astype(np.float64) / float(1 << quant_bits)


def overflow_fraction(x: np.ndarray, quant_bits: int, modulus: int) -> float:
    """Fraction of coordinates of ``x`` that fall outside the representable band."""
    limit = float(modulus) / 2.0 / float(1 << quant_bits)
    arr = np.asarray(x, dtype=np.float64)
    return float(np.mean(np.abs(arr) >= limit)) if arr.size else 0.0


def max_abs(arrays: Sequence[np.ndarray]) -> float:
    """Max absolute value across a list of arrays (0.0 for an empty list)."""
    return max((float(np.max(np.abs(a))) for a in arrays if a.size), default=0.0)


def l2(arrays: Sequence[np.ndarray]) -> float:
    return float(np.sqrt(sum(float(np.sum(np.asarray(a, dtype=np.float64) ** 2)) for a in arrays)))


def is_maskable(arr: np.ndarray) -> bool:
    """True for the float entries a mask can be added to and removed from cleanly.

    A ``state_dict`` is not all gradients. BatchNorm contributes an int64
    ``num_batches_tracked``, and Hugging Face models carry integer buffers such as
    ``position_ids``. Adding a float mask to one of those truncates on the cast back, so the
    masks no longer cancel and the aggregate is silently wrong. These entries carry counters
    rather than learned information, so both defenses leave them alone.
    """
    return np.issubdtype(np.asarray(arr).dtype, np.floating)
