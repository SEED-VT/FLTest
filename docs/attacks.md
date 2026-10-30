# Attacks

Attacks are hook plugins (`fltest/attacks/`) that subclass `ThreatModelBaseClass`. Declare
them in a config:

```yaml
attacks:
  - {name: <attack>, params: {...}, target_clients: [0, 1]}   # target_clients optional (default: all)
```

Multiple attacks compose. `target_clients` restricts which clients are adversarial.

`backdoor`, `little_is_enough_backdoor`, `little_is_enough_cropped_backdoor`, and `dlg`
operate on pixels, so they apply only
to image datasets and raise a clear error on a text run. `label_flip`, `sign_flip`, `gaussian`, `model_replacement`, and
`little_is_enough` work on labels or updates, so they apply to either modality.

## Catalog

| Name | Type | Hook(s) | Key params |
|------|------|---------|-----------|
| `label_flip` | data poisoning | `before_client_train` | `shift` (default 1), `mapping` |
| `gaussian` | model poisoning (naive) | `after_client_train` | `sigma` (0.1) |
| `sign_flip` | model poisoning | `after_client_train` | `scale` (1.0) |
| `model_replacement` | model poisoning (targeted) | `after_client_train` | `scale` (automatic), `target_round` |
| `little_is_enough` | model poisoning (Byzantine, evasive) | `before_aggregate` | `z` (automatic), `target_clients` (required) |
| `little_is_enough_cropped_backdoor` | model poisoning (targeted, evasive) | `on_data_distribute`, `before_aggregate`, `after_round` | `z` (automatic), `target_label` (0), `patch_size` (5), `target_clients` (required) |
| `little_is_enough_backdoor` | model poisoning (targeted, evasive) | `on_data_distribute`, `before_aggregate`, `after_round` | `z` (automatic), `alpha` (1.0), `epochs` (5), `samples` (1000), `target_label` (0), `patch_size` (5), `target_clients` (required) |
| `backdoor` | data poisoning (targeted) | `before_client_train`, `after_round` | `target_label` (0), `infection_rate` (0.3), `patch_size` (4), `patch_value` (1.0) |
| `dlg` | privacy (gradient inversion) | `before_client_train` (+ `before_aggregate` in shared_update mode) | `target_client`, `target_round`, `num_images`, `iters`, `source` |
| `membership_inference` | privacy (inference) | `on_data_distribute`, `after_round` | `target_client` (0), `max_samples` (512) |

## How each works

**`label_flip`** — wraps the attacker's loader and relabels each batch
(`y → (y+shift) % num_classes`, or a fixed `{src: dst}` mapping). A classic robustness
attack; weak alone (one of the "naive" attacks the project flags).

**`gaussian`** — adds zero-mean Gaussian noise to the attacker's update
(`u' = u + N(0, sigma²)`). Naive Byzantine attack; useful as a baseline.

**`sign_flip`** — reflects the update around the global model and scales it
(`u' = g − scale·(u − g)`), pushing aggregation in the opposite direction.

**`model_replacement`** — boosts a malicious local model around the current global model
(`u' = g + scale·(u − g)`) so the malicious delta survives aggregation. This implements
the train-and-scale attack from Bagdasaryan et al., [*How To Backdoor Federated
Learning*](https://proceedings.mlr.press/v108/bagdasaryan20a.html). Compose it after
`backdoor` to boost a locally learned trigger, and set `target_round` for a single-shot
attack. Without an explicit `scale`, FLTest uses `num_clients / num_attackers`, which is
the replacement factor for full-participation, equal-weight FedAvg.

The hook interface exposes the local model, global model, client identity, and round, so
the attack itself is backend-neutral on reference and Flower. It does not expose the
round's total sample weight to a client, however. Automatic scaling is therefore only an
estimate when clients have unequal sample weights; pass the exact factor explicitly in
that case. NVFlare does not run client-side hooks and cannot apply this attack.

The runnable comparison uses one of four clients as the attacker and boosts its backdoored
model only in round 3, at an automatic scale of 4.0. Replacement takes attack success from
0.2930 to 1.0000 on the reference backend and from 0.2724 to 0.9892 on Flower, costing
0.0196 and 0.0361 of clean accuracy:

| Backend | Attack | ASR | Accuracy |
|---------|--------|----:|---------:|
| reference | backdoor only | 0.2930 | 0.8926 |
| reference | model replacement | 1.0000 | 0.8730 |
| Flower | backdoor only | 0.2724 | 0.8867 |
| Flower | model replacement | 0.9892 | 0.8506 |

**`little_is_enough`** — a Byzantine attack that measures the honest population before
perturbing it, from Baruch et al., [*A Little Is Enough: Circumventing Defenses For
Distributed Learning*](https://proceedings.neurips.cc/paper/2019/hash/ec1c59141046cd1866bbbcdfb6ae31d4-Abstract.html).
`gaussian` and `sign_flip` perturb an update without reference to the honest ones, so a
perturbation big enough to matter is also an outlier — exactly what `krum`, `median` and
`trimmed_mean` filter. This attack instead sets every attacker's update to the benign mean
shifted by a fraction of the benign standard deviation, per coordinate:

```
u' = mu - z * sigma
```

With `n` participants and `m` attackers, `s = floor(n/2 + 1) - m` honest workers are needed
to complete a majority, and the default `z` is the largest value with
`Phi(z) < (n - m - s) / (n - m)` — the most an attacker can move while still sitting inside
the range a majority of honest workers occupy. The attack runs at `before_aggregate`, where
FLTest exposes every submission, and rewrites only the attackers' entries; attacks attach
before defenses, so a robust aggregator in the same config still sees the crafted updates.
`target_clients` is required, because the crafted update is built from the *other* clients'
submissions. NVFlare does not emit `before_aggregate` and cannot run this attack.

**Reading the outcome.** Accuracy alone is a poor success signal for an untargeted Byzantine
attack, so the attack scores itself at `after_aggregate`. It records the `z` it asked for,
`little_is_enough_drift` (how far the aggregate actually landed from the benign mean, in
benign standard deviations) and `little_is_enough_absorption` (`drift / z` — the share of the
requested shift the defense conceded). The anchor for reading absorption is plain FedAvg,
which concedes exactly the attackers' share of the round's total sample weight — `m/n` when
every client holds an equal shard, and their data share otherwise. A defense scoring above
its own FedAvg line is doing worse than no defense at all.

Measured over one aggregation round on a controlled population of ten clients, four of them
adversarial, so the numbers isolate what each rule concedes rather than how training then
unfolds (the `rules-*` runs in `little_is_enough.yaml` are the trained version):

| Aggregation | `z = 1.0` | `z = 1.5` | `z = 3.0` |
|-------------|----------:|----------:|----------:|
| FedAvg (no defense), 4 of 10 attackers | 0.40 | 0.40 | 0.40 |
| `krum` | 1.00 | 1.00 | 0.00 |
| `median` | 0.85 | 0.69 | 0.36 |
| `trimmed_mean`, `trim: 2` | 0.63 | 0.57 | 0.46 |

Krum concedes everything while it accepts the craft, because it returns one selected update
and that update is the attacker's — strictly worse than averaging. It drops to zero the
moment `z` puts the craft outside the honest spread. The coordinate-wise rules never fully
accept or reject: they concede a share that *falls* as `z` grows, because pushing an already
extreme value further does not move a median.

The `rules-*` runs in `little_is_enough.yaml` train at `z = 1.5` for four rounds. Over the
same five seeds:

| Run | Accuracy | Absorption |
|-----|---------:|-----------:|
| no attack, FedAvg | 0.8283 ± 0.0142 | – |
| attack, FedAvg (no defense) | 0.7939 ± 0.0225 | 0.4000 ± 0.0000 |
| attack, `krum` | 0.7582 ± 0.0332 | 1.0000 ± 0.0000 |
| attack, `median` | 0.7783 ± 0.0243 | 0.6892 ± 0.0019 |
| attack, `trimmed_mean`, `trim: 2` | 0.7873 ± 0.0225 | 0.5616 ± 0.0017 |

**Every robust rule here concedes more than plain averaging does.** Against this attack the
protection ordering runs backwards: no defense (0.40) is the most resistant, then
`trimmed_mean` (0.56), then `median` (0.69), and `krum` (1.00) hands over the whole model.

Note how differently the two columns behave. Absorption is reproducible to the fourth decimal
across seeds; accuracy varies by ±0.02 to ±0.03, which is wider than the gaps between the
rows. So the absorption ordering is a result, and the matching accuracy ordering is not — it
may well be coincidence at this scale. Krum's accuracy spread is the widest of all (±0.0332),
which is Krum itself rather than the attack: even when it defends successfully it keeps one
update and discards nine, so the round's model is whichever client it happened to pick.

The runnable sweep puts four attackers among ten clients behind Krum for eight rounds, over
five seeds (786, 101, 202, 303, 404):

| Run | `z` | Accuracy | Absorption | Krum's pick |
|-----|----:|---------:|-----------:|-------------|
| no attack | – | 0.8559 ± 0.0222 | – | an honest client |
| `little_is_enough` (automatic) | 0.4307 | 0.8775 ± 0.0161 | 1.00 | the attacker |
| `little_is_enough` | 1.5 | 0.8473 ± 0.0244 | 1.00 | the attacker |
| `little_is_enough` | 3.0 | 0.8303 ± 0.0691 | 0.03 | an honest client |
| `gaussian`, `sigma: 0.5` | – | 0.8303 ± 0.0691 | – | an honest client |

**Evasion is reliable here; damage is not.** Absorption is 1.00 at both `z = 0.43` and
`z = 1.5` in every seed — Krum takes the crafted update whole, every round, deterministically.
That is the attack working exactly as the paper describes, and it is the result to quote.

Accuracy is a different story. At `z = 1.5` the attack costs 0.9 points against a baseline
whose own seed-to-seed spread is ±0.022, so at this scale getting past Krum buys the attacker
nothing measurable. Do not read the middle row as damage. An earlier single-seed run of this
same config showed a 4.6-point drop at `z = 1.5`; four more seeds showed that was the seed,
not the attack.

Two findings do survive the seeds. The automatic `z` of 0.43 makes accuracy *rise*, in four
of five seeds: the craft sits so close to the benign mean that Krum selecting it beats Krum's
usual pick of one client's noisy update, so the attacker is invisible and actively unhelpful
to itself. And at `z = 3.0` the craft has left the honest spread, Krum rejects it (absorption
0.03), and the run becomes identical to the filtered `gaussian` baseline — not similar,
identical, in all five seeds, because both are filtered and Krum falls back to the same
honest client.

The lesson for using this attack is that accuracy is too noisy to evaluate it at this scale.
Score it with `little_is_enough_absorption`, which is deterministic and separates "the defense
filtered me" from "the defense took the bait."

The automatic `z` is small whenever attackers are a small fraction of the round: it is 0
below roughly `n/2` attackers and reaches the paper's headline 1.43 only at 24 attackers out
of 50. Treat it as the stealth ceiling, not as a recommended strength, and sweep `z`
(`attack_strength` in [metamorphic testing](metamorphic-testing.md)) to find where a defense
actually breaks.

**`little_is_enough_backdoor`** — the same paper's backdoor variant (Section 3.4,
Algorithm 4), and the version of "A Little Is Enough" the FLDetector paper evaluates. It
spends the `z`-deviation budget on a goal instead of a fixed shift: starting from the
benign mean `mu`, the attacker trains on `samples` of its own images with the upper-left
`patch_size` x `patch_size` pixels set to maximal intensity and relabelled `target_label`,
for `epochs` passes, then clamps every parameter into `mu ± z·sigma` and submits the result
from every attacker. The defaults are the paper's pattern experiment: 1000 images, a 5x5
trigger, label 0, five passes. It records `attack_success_rate` every round, the share of
triggered test images (true label other than the target) sent to `target_label`.

`alpha` weights the paper's Equation 3, `alpha·l_backdoor + (1 - alpha)·l_delta`. Equation
4's `l_delta` measures each parameter's distance from `mu` in units of `z·sigma`, and
honest clients agree so closely that this term's curvature, `1/(z·sigma)^2`, makes plain SGD
diverge; it is minimised exactly with a proximal step instead. Summed over every parameter
it also outweighs the backdoor loss so heavily that the optimum stays at `mu`: on MNIST with
the MLP, `alpha = 0.2` leaves the crafted update identical to the benign mean. The default
`alpha = 1` lets the clamp alone keep the update inside the honest range, which is the
clamp's job, and the backdoor then builds up round after round. Median over 30 rounds at 28%
attackers reaches ASR 0.49 at `z = 0.74` while clean accuracy stays at 0.91; at the paper's
`z = 0.2` it reaches only 0.03 in that time.

**`little_is_enough_cropped_backdoor`** — the same attack as the FLDetector paper describes
running it: each malicious update "computed following the Scaling Attack", which the
attacker then "crops" into range. Every round, each attacker trains its own model from the
global model on its own images plus a trigger-stamped duplicate of each, labelled
`target_label` (the Scaling attack's augmentation, scaling factor 1), with the run's
`client_epochs`, `client_lr`, and `client_batch_size`. It then clamps every parameter into
the honest `mu ± z·sigma`. `little_is_enough_backdoor` trains one update from the honest
mean and gives it to every attacker; this one gives each attacker its own, anchored to its
own poisoned training. That is the difference a consistency detector such as `fldetector`
reads. Trigger, target label, and `attack_success_rate` are as for
`little_is_enough_backdoor`.

**`backdoor`** — the attacker stamps a bright patch on a fraction (`infection_rate`) of its
images and relabels them to `target_label`; the global model learns
*trigger ⇒ target_label*. At each round end it measures **attack success rate (ASR)** — the
fraction of a triggered test set predicted as the target — and records it as a metric. This
is the headline robustness signal; pair it with a robust-aggregation defense to see ASR drop
(see [Defenses](defenses.md)).

**`membership_inference`** — asks whether a given record was in a client's training data,
which is the canonical privacy attack the proposal cites and the one Pitfall-1 says
evaluations skip. It models an honest-but-curious server that sees the global model each
round. The score is the per-sample loss, following Yeom et al., since a model assigns lower
loss to data it trained on. Members are the target client's training data and non-members
are the held-out test set. It records `membership_inference_auc`, where 0.5 means no
leakage and 1.0 means members and non-members separate perfectly, alongside
`membership_loss_gap`. No shadow model is needed, and because it reads only losses it
applies to text as readily as to images.

For example, `examples/configs/membership_inference.yaml` runs the same overfitted setup
twice. Undefended it reaches AUC 0.67, and clipping with Gaussian noise takes it to 0.50,
which is chance, for about six points of accuracy.

**`dlg`** — Deep Leakage from Gradients: reconstructs a victim client's private batch by
optimizing a dummy batch so its gradient matches the victim's. Records `reconstruction_mse`,
`reconstruction_psnr`, and `label_recovery`. Use `model_name: ConvNet` (smooth activations)
on `device: cpu`. Two threat sources:

- `source: gradient` (default) — reconstruct from the *raw per-step gradient*; demonstrates
  pure invertibility.
- `source: shared_update` — reconstruct from the *uploaded (post-defense) update*; faithful
  only under single-step (FedSGD) training.

## Examples

```yaml
# label flip on two clients
attacks: [{name: label_flip, params: {shift: 1}, target_clients: [0, 1]}]
```

```yaml
# backdoor measured by ASR
attacks: [{name: backdoor, params: {target_label: 0, infection_rate: 0.8, patch_size: 5}, target_clients: [0, 1]}]
```

```yaml
# boost one backdoored client in round 3; scale defaults to num_clients
attacks:
  - {name: backdoor, params: {target_label: 0, infection_rate: 0.8}, target_clients: [0]}
  - {name: model_replacement, params: {target_round: 3}, target_clients: [0]}
```

```yaml
# privacy attack
model_name: ConvNet
dataset: cifar10
attacks: [{name: dlg, params: {target_client: 0, target_round: 1, iters: 300, source: gradient}}]
```

```yaml
# Byzantine attack sized to the honest spread; sweep z to find where the defense breaks
defenses: [{name: krum, params: {num_byzantine: 4}}]
attacks: [{name: little_is_enough, params: {z: 1.5}, target_clients: [0, 1, 2, 3]}]
```

Runnable: `examples/configs/attack_label_flip.yaml`, `model_replacement.yaml`,
`little_is_enough.yaml` (strength sweep against Krum, the same attack against each
aggregation rule, and against FLDetector), `dlg.yaml`.

To add your own attack, see **[Port your attacks & defenses](extending.md)**.
