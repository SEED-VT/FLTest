"""MPC aggregation as a loadable hook file — and what that costs.

    export FLTEST_HOOKS=examples/hooks/def_mpc_aggregation
    fltest run examples/configs/dlg.yaml

The protocol itself is unaffected: ``mpc_aggregation`` only uses ``before_aggregate``, which
runs on the driver, so unlike ``secure_aggregation`` (which masks client-side, inside Ray
workers) nothing about this placement is fragile. The aggregate is correct and every metric
— ``mpc_agg_max_abs_error``, ``mpc_overflow_rate``, ``mpc_dropouts`` — still reaches the
report through ``ctx.record``.

What a hook file cannot do is *declare* itself, and four things hang off that declaration:

1. The pitfall checker reads ``config.defenses``. Loaded this way the defense is invisible to
   it, so a ring too small for the summed aggregate, too few ``quant_bits``, or a non-zero
   ``dropout_rate`` all go unflagged. Those failures are silent by nature — overflow wraps
   around and yields a finite, plausible-looking aggregate — so this is the check that
   matters most.
2. The ``secagg_lossless`` metamorphic relation sweeps ``defense.seed`` in the config. With
   no defense entry to vary it raises rather than running.
3. The report records ``aggregation: fedavg`` and an empty defense list, and the run_id
   hashes identically to an undefended run, so the two cannot be told apart afterwards.
4. Hook order is fixed: ``build_hook_runner`` attaches FLTEST_HOOKS last. Any declared robust
   aggregator therefore runs first and collapses the updates to a single entry — at which
   point pairwise masking has no pair to work with and silently does nothing. Declared in
   ``defenses:`` you choose the order; here you cannot.

Use this to prototype. For anything whose result you intend to report, declare it:

    defenses:
      - {name: mpc_aggregation, params: {quant_bits: 16, modulus: 4294967296}}
"""

from fltest.core import hooks
from fltest.defenses.mpc_aggregation import MPCAggregationDefense

# Frozen here: a hook file has no channel to the YAML, so these cannot be swept by the
# config fuzzer or varied by a metamorphic relation.
_defense = MPCAggregationDefense(quant_bits=16, modulus=1 << 32, dropout_rate=0.0, seed=1)


@hooks.before_aggregate
def mpc_aggregate(ctx):
    _defense.before_aggregate(ctx)
