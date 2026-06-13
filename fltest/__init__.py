"""FLTest: a testbed for evaluating privacy & robustness of Privacy-Preserving FL.

Public surface:
* ``fltest.core``      — hooks, HookContext, config (RunSpec/TestConfig), orchestrator.
* ``fltest.frameworks``— FL backends behind one ``run_simulation`` abstraction.
* ``fltest.attacks`` / ``fltest.defenses`` / ``fltest.metrics`` — composable hook plugins.
* ``fltest.testing``   — differential and metamorphic testing engines.
* ``fltest.pitfalls``  — pitfall checker + counter-experiment recommender.
"""

from fltest.core import hooks

__all__ = ["hooks"]
__version__ = "0.2.0"
