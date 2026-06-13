# FLTest

**A testbed for evaluating the privacy and robustness of Privacy-Preserving Federated
Learning (PPFL).**

FLTest gives you *software-defined control and visibility* into FL testing. From a single
YAML config you can:

- run the **same experiment across multiple FL frameworks** (a built-in reference oracle,
  Flower, NVFlare) behind one `run_simulation()` abstraction;
- inject **attacks and defenses as composable hooks** that work unchanged across backends;
- apply **differential testing** (frameworks must agree) and **metamorphic testing**
  (input→output relations must hold);
- run a **pitfall checker** that flags the common FL-evaluation mistakes and recommends
  counter-experiments;
- **fuzz** a config (any list-valued knob is expanded into a grid of runs).

This is the implementation of the NSF PDaSP Track-3 *FLTEST* proposal.

## Why it exists

A survey of 50 FL robustness papers found highly inconsistent experimental setups —
MNIST-only, IID-only, naive attacks, only global accuracy reported. These choices inflate
privacy/robustness claims and make results hard to compare. FLTest makes a rigorous setup
the default and *checks* for those pitfalls automatically.

## The 60-second mental model

```
test_conf.yaml ─▶ fuzzer ─▶ many RunSpecs ─▶ adapter.run_simulation() ─▶ RunResult ─▶ tests + reports
                   (grid)     (one per          (reference / flwr /        (metrics)     (differential,
                              framework×knob)     nvflare)                                metamorphic,
                                                                                          pitfalls)
```

Everything an experiment does — poison data, perturb updates, reconstruct gradients, record
a metric — is a **hook** sharing one mutable `HookContext`. A hook written once runs across
every framework, and several hooks compose on the same run.

## Where to go next

- New here? **[Installation](installation.md)** → **[Quickstart](quickstart.md)** →
  **[Approach walkthrough](walkthrough.md)**.
- Want the internals? **[Concepts & internals](concepts.md)** and **[Architecture](ARCHITECTURE.md)**.
- Setting up experiments? **[Configuration reference](configuration.md)** and
  **[Config fuzzing](fuzzing.md)**.
- Using/adding data, attacks, defenses? **[Datasets](datasets.md)**, **[Attacks](attacks.md)**,
  **[Defenses](defenses.md)**, **[Metrics](metrics.md)**.
- Bringing your own technique? **[Port your attacks & defenses](extending.md)**.
