# FLTest

### A Testbed for Enhancing Privacy and Robustness of Federated Learning Systems

FLTest is an open, community testbed for **evaluating the privacy and robustness of
Privacy-Preserving Federated Learning (PPFL)**. From a single YAML config it runs the same
experiment across multiple FL frameworks, injects attacks and defenses as composable hooks,
and applies **differential** and **metamorphic** testing plus a **pitfall checker** — giving
researchers *software-defined control and visibility* into FL evaluation.

> Developed under the **NSF PDaSP Program (Track 3), Award #2452817-19**.

---

## The problem

A survey of 50 federated-learning robustness papers found wildly inconsistent experimental
setups — MNIST-only, IID-only data, naive attacks, and only global accuracy reported. These
choices systematically **over-estimate** privacy and robustness and make results hard to
reproduce or compare. There is no standardized way to *test* whether a PPFL technique is
actually private and robust.

## What FLTest does

FLTest streamlines the often complex and labor-intensive evaluation of privacy and
robustness in PPFL systems. From a single configuration file it orchestrates end-to-end
federated-learning experiments across multiple frameworks through one common
`run_simulation()` abstraction, so the same setup can be run — and directly compared — on a
dependency-light reference oracle, Flower, and NVFlare. Every adversarial and protective
behaviour is expressed as a composable hook sharing one context: attacks, defenses, and
metric listeners are written once and run unchanged across every backend, and several can be
combined on a single run.

On top of this orchestration, FLTest brings the discipline of software testing to PPFL.
Differential testing checks that frameworks given the same configuration agree within
tolerance, so an unexplained divergence points to an implementation bug. Metamorphic testing
verifies that input–output relations which *should* hold actually do — for example, doubling
the number of clients on IID data should not reduce accuracy, and strengthening an attack
should not improve it. A pitfall checker, drawing on a continuously updated catalog of known
evaluation pitfalls, detects configuration issues and overlooked vulnerabilities early, and a
recommendation engine proposes counter-experiments that strengthen a weak evaluation. A
configuration fuzzer expands any list-valued knob into a grid of experiments, allowing broad
coverage of models, datasets, distributions, and attack settings from a compact config.

## Project goals

The project designs, develops, and sustains FLTest as a comprehensive, standardized testbed
that automates privacy and robustness evaluation for PPFL systems, lowering the barrier to
rigorous federated-learning research for both novice and expert users. It aims to make
results reproducible and portable by enabling cross-framework and cross-configuration
comparison across Flower, NVFlare, IBM FL, and other frameworks, and to counter the common
over-estimation of privacy and robustness claims by detecting and remediating evaluation
pitfalls automatically. FLTest treats advanced attacks and defenses — including
gradient-inversion privacy attacks, backdoors, robust aggregation, and differential
privacy — as first-class, composable components, and it is built to grow a sustainable
open-source community around trustworthy FL evaluation, integrated with existing FL
frameworks and NSF cyberinfrastructure.

## Objectives

Concretely, FLTest delivers an automated test-orchestration module that generates diverse,
fault-revealing federated deployments from a single configuration; a pitfall checker backed
by a catalog of known FL-evaluation pitfalls that is continuously updated to reflect new
research; and a recommendation engine that turns detected pitfalls into actionable
counter-experiments. It provides support for privacy-preserving techniques such as
differential privacy, secure aggregation, and robust aggregation, paired with privacy- and
robustness-specific evaluation metrics — including personalized, per-client evaluation that
exposes representation disparity a single global number would hide. Finally, it is designed
for deployment at scale on FL frameworks and cloud testbeds so that techniques can be
validated under realistic, real-world conditions.

---

## Investigators

| Principal Investigator | Institution |
|------------------------|-------------|
| [**Ali Anwar**](https://chalianwar.github.io/) | University of Minnesota |
| [**Muhammad Ali Gulzar**](https://people.cs.vt.edu/~gulzar/) | Virginia Tech |
| [**Fatima Anwar**](https://people.umass.edu/fanwar/) | University of Massachusetts Amherst |

## Sponsor & partner institutions

<p align="center" style="display:flex; align-items:center; justify-content:center; gap:48px; flex-wrap:wrap;">
  <img src="assets/logos/nsf.png" alt="U.S. National Science Foundation" style="height:72px; width:auto;">
  <img src="assets/logos/umn.png" alt="University of Minnesota" style="height:48px; width:auto;">
  <img src="assets/logos/vt.png" alt="Virginia Tech" style="height:56px; width:auto;">
  <img src="assets/logos/umass.png" alt="University of Massachusetts Amherst" style="height:40px; width:auto;">
</p>

## Acknowledgement

This material is based upon work supported by the **U.S. National Science Foundation** under
the **Privacy-preserving Data Sharing in Practice (PDaSP) program, Track 3 — Usable Tools and
Testbeds for Confidential Data Sharing**, **Award #2452817-19**. The PDaSP program is supported
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
