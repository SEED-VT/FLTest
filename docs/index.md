# FLTest

### A Testbed for Enhancing Privacy and Robustness of Federated Learning Systems

FLTest is an open, community testbed for **evaluating the privacy and robustness of
Privacy-Preserving Federated Learning (PPFL)**. From a single YAML config it runs the same
experiment across multiple FL frameworks, injects attacks and defenses as composable hooks,
and applies **differential** and **metamorphic** testing plus a **pitfall checker** — giving
researchers *software-defined control and visibility* into FL evaluation.

> Developed under the **NSF PDaSP Program (Track 3), Award #2452819**.

---

## The problem

A survey of 50 federated-learning robustness papers found wildly inconsistent experimental
setups — MNIST-only, IID-only data, naive attacks, and only global accuracy reported. These
choices systematically **over-estimate** privacy and robustness and make results hard to
reproduce or compare. There is no standardized way to *test* whether a PPFL technique is
actually private and robust.

## What FLTest does

- **One abstraction, many frameworks.** A single `run_simulation()` adapter runs the same
  experiment on a reference oracle, **Flower**, and **NVFlare** — and you can compare them.
- **Everything is a hook.** Attacks, defenses, and metrics are composable plugins sharing one
  context, so a technique written once runs unchanged across every backend.
- **Differential testing.** Frameworks must agree on the same config within tolerance;
  divergence points to implementation bugs.
- **Metamorphic testing.** Input→output relations must hold (e.g. doubling clients shouldn't
  drop accuracy; stronger attacks shouldn't raise it).
- **Pitfall checker + recommender.** Flags the six common FL-evaluation pitfalls and emits
  copy-pasteable counter-experiments.
- **Config fuzzer.** Any list-valued knob (`dataset: [mnist, cifar10]`) expands into a grid.

## Project goals

1. Provide a **standardized, automated testbed** for privacy and robustness evaluation of
   PPFL systems, lowering the barrier to rigorous FL research.
2. Enable **cross-framework** and **cross-configuration** comparison so results are
   reproducible and portable across Flower, NVFlare, IBM FL, and others.
3. **Detect and remediate evaluation pitfalls** automatically, countering the
   over-estimation of privacy/robustness claims.
4. Support **advanced attack and defense models** (gradient-inversion privacy attacks,
   backdoors, robust aggregation, differential privacy) as first-class, composable plugins.
5. Grow a **sustainable open-source community** around trustworthy FL evaluation, integrated
   with existing FL frameworks and NSF cyberinfrastructure.

## Objectives

- **Automated test orchestration** that generates diverse, fault-revealing FL deployments
  from a single configuration.
- **A pitfall checker** with a continuously-updated catalog of known FL-evaluation pitfalls.
- **A recommendation engine** that proposes counter-experiments to strengthen an evaluation.
- **PPFL technique support** (DP, secure aggregation, robust aggregation) with tailored
  privacy/robustness metrics, including personalized (per-client) evaluation.
- **Deployment at scale** on FL frameworks and cloud testbeds for real-world evaluation.

---

## Investigators

| Principal Investigator | Institution |
|------------------------|-------------|
| [**Ali Anwar**](https://chalianwar.github.io/) | University of Minnesota |
| [**Muhammad Ali Gulzar**](https://people.cs.vt.edu/~gulzar/) | Virginia Tech |
| [**Fatima Anwar**](https://people.umass.edu/fanwar/) | University of Massachusetts Amherst |

## Sponsor & partner institutions

<p align="center">
  <img src="assets/logos/nsf.png" alt="U.S. National Science Foundation" height="80">
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="assets/logos/umn.png" alt="University of Minnesota" height="64">
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="assets/logos/vt.png" alt="Virginia Tech" height="72">
  &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="assets/logos/umass.png" alt="University of Massachusetts Amherst" height="64">
</p>

## Acknowledgement

This material is based upon work supported by the **U.S. National Science Foundation** under
the **Privacy-preserving Data Sharing in Practice (PDaSP) program, Track 3 — Usable Tools and
Testbeds for Confidential Data Sharing**, **Award #2452819**. The PDaSP program is supported
by the NSF together with its co-sponsors (U.S. Department of Transportation, Intel, NIST, and
Broadcom). Any opinions, findings, and conclusions or recommendations expressed in this
material are those of the authors and do not necessarily reflect the views of the National
Science Foundation or its co-sponsors.

Program information: [pdasp.net/projects](https://pdasp.net/projects/).

## Contact

Reach the investigators via their pages:

- [Ali Anwar](https://chalianwar.github.io/) — University of Minnesota
- [Muhammad Ali Gulzar](https://people.cs.vt.edu/~gulzar/) — Virginia Tech
- [Fatima Anwar](https://people.umass.edu/fanwar/) — University of Massachusetts Amherst

For bugs, feature requests, and contributions, please use the
[GitHub issue tracker](https://github.com/SEED-VT/FLTest/issues).

---

## Get started

<div class="grid cards" markdown>

- **[Install](installation.md)** — set up the isolated environment.
- **[Quickstart](quickstart.md)** — run your first experiment.
- **[Approach walkthrough](walkthrough.md)** — how a config becomes a tested result.
- **[Worked example](worked-example.md)** — exhaustively testing attacks & defenses.
- **[Configuration](configuration.md)** & **[Fuzzing](fuzzing.md)** — every knob, and the grid.
- **[Port your attacks & defenses](extending.md)** — bring your own technique.

</div>
