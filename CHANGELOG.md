# Changelog

Every release of FLTest is recorded here. Versions follow [semantic
versioning](https://semver.org). The patch number changes for a fix and the minor number
for new capability that leaves existing configs working. The major number changes when the
configuration schema or the plugin API breaks.

## 0.9.0

**Online FLDetector defense.** Added a hook-based detector for the reference and Flower
backends. It compares each client's current model delta against a history-based L-BFGS
prediction, averages normalized inconsistency over a configurable window, and uses gap
statistics plus two-cluster k-means to identify suspicious clients. Detected submissions
are filtered before aggregation; `before_round` excludes those clients in later rounds.
This online variant does not restart training from the initial model as in the paper.
When paired with Krum, trimmed mean, or median, list `fldetector` first so filtering
precedes the aggregation rule. Flower now carries server-side aggregation-hook metrics
into round history.

## 0.8.0

**Before-round client selection.** The `before_round` hook now receives all eligible
client IDs in `ctx.selected_clients` on reference and Flower. A hook can replace them with
a nonempty, duplicate-free list or tuple to choose which clients train in that round.
Flower now emits `before_round` before dispatch rather than after fit results arrive. It
looks up stable partition IDs through client properties only for selective rounds; ordinary
full-participation runs do not add that request. Custom Flower clients must report a valid
`cid` property when selection is used. NVFlare's replayed round hooks remain observational.

## 0.7.0

**Aggregate result override.** On reference and Flower, `on_aggregate` may now replace
`ctx.new_global_state` with the model to use for evaluation and the next round. The
`after_aggregate` hook observes that committed model. Leaving the field unchanged keeps
the existing weighted-average behavior, and setting it to `None` also retains the
computed aggregate. NVFlare still does not support an aggregation-result override.

## 0.6.0

**Identified aggregation context.** The `before_aggregate` hook now includes each received
client's stable ID, update, and sample count in `ctx.client_submissions` on reference and
Flower. Flower also provides the current global model in `ctx.global_state`. This lets
history-based defenses associate updates with the same client across rounds without
assuming an arrival order. The existing `ctx.updates_and_weights` input and built-in
robust defenses are unchanged. Custom Flower clients without a reported ID still
aggregate normally and expose `client_id=None`. NVFlare still does not emit
`before_aggregate`.

## 0.5.0

**Model replacement.** Added the train-and-scale model-replacement attack. It boosts a
malicious client's locally trained delta around the current global model, and composes
with the existing backdoor attack so a trigger learned locally survives FedAvg. A
single-shot target round, explicit scaling, and automatic equal-weight scaling across one
or more colluding clients are supported on the reference and Flower backends.

The implementation also served as a hook-interface audit. The existing client hook exposes
the local model, global model, client identity, and round, which is enough for the attack.
It does not expose the selected round's total sample weight, so automatic scaling cannot be
exact for unequal-weight FedAvg; an explicit scale is required there. NVFlare still does
not support client-side hooks.

**Fixed.** Flower's client hook contexts always reported round zero because the server did
not include `server_round` in fit configuration. It now sends the round already consumed
by `FlowerClient`, so targeted-round attacks behave consistently across reference and
Flower.

## 0.4.8

**README.** The highlights advertised text datasets and Hugging Face models, but the
install block never mentioned the `[hf]` extra that provides them, so a reader following
the README could not reach a feature it promised. It is listed now, with a pointer to what
an Intel Mac pins.

The command list also predated most of the examples. It now covers the attack and defense
matrix, both privacy attacks, FEMNIST partitioned by writer, and federated text, and its
differential example is the CIFAR-10 three-way rather than the MNIST one, which cannot
catch a backend that loses the channel count.

## 0.4.7

**Fixed.** A run that failed because transformers had disabled its PyTorch backend reported
`ImportError:` and nothing else. transformers raises a message beginning with a blank line,
so the first line of the recorded error held only the exception name. The run matrix now
carries the next line with content up beside it.

**Changed.** Building an `hf:` model checks that transformers still has its PyTorch backend
before it tries. transformers reports that PyTorch "was not found" in this case, which is
misleading, since torch is installed and every built-in model trains with it. FLTest names
the real cause, which is the version floor, and the fix, which is the capped `[hf]` extra.

Also documented that torch 2.2.2 needs `numpy<2`.

## 0.4.6

**Fixed.** A three-way differential printed 165 lines, 143 of them NVFlare INFO and WARNING
records, even without `-v`. `logging.disable` only affects the process that calls it, and
NVFlare runs each client in its own process, so the suppression never reached them. The
same held for Ray workers, which repeated every Hugging Face import message once per
worker because log deduplication was switched off.

Quiet mode now sets `FL_LOG_LEVEL`, `TRANSFORMERS_VERBOSITY`, and `RAY_DEDUP_LOGS` in the
environment, which a child process does inherit. The same run now prints 22 lines with no
INFO or WARNING records, and `-v` still produces the full 415.

## 0.4.5

**Fixed.** The `[hf]` extra now pins `transformers<5`. transformers 5 requires torch 2.5 or
later and silently disables PyTorch below it, so `hf:` models and text datasets stopped
working while tokenizers kept loading. PyTorch ships no macOS x86_64 wheel past 2.2.2,
which means an Intel Mac cannot satisfy that floor at any version. The cap keeps the extra
working on both architectures.

`environment.yml` also referred to a `[pfl]` extra that no longer exists, and now points at
`[hf]` instead.

## 0.4.4

**Fixed.** The built-in datasets used bare Hugging Face ids, so `mnist`, `fashion_mnist`,
and `cifar10` were passed to the Hub as-is. huggingface-hub 1.16 removed that form, and
a fresh environment installs a later version, so any of those datasets failed with
`HfUriError` on a machine without a warm cache. The ids are now namespaced as
`ylecun/mnist`, `zalando-datasets/fashion_mnist`, and `uoft-cs/cifar10`, which works on
old and new versions alike. The short names in a config are unchanged.

The failure only appeared on a cold cache, because a machine that had already downloaded
the dataset answered from disk and printed a note about the Hub lookup failing. A test now
asserts every built-in id is namespaced, since a run on a warm machine cannot catch this.

A Hub id given directly in a config is checked too. A bare name that is not a built-in now
explains that the id needs a namespace, instead of surfacing the Hub's own error.

## 0.4.3

**Worked example.** The exhaustive attack and defense matrix moves from MNIST with an MLP
to CIFAR-10 with LeNet, and gains two privacy scenarios. Ten balanced classes make a
collapsed model obvious, since chance sits at 0.10.

The new numbers carry findings the MNIST version could not show. `norm_clip` at 0.5 reports
the worst attack success rate in the matrix, 0.9924, while its accuracy falls to 0.1094.
Clipping that hard stopped the model learning and it collapsed to predicting the attacker's
target label, which is a constant predictor scoring near 1.0 on a triggered test set. That
makes the case that attack success rate means nothing read alone.

Membership inference returns 0.4831, which is chance, because this model underfits and has
memorized nothing to expose. The dedicated example reaches 0.67. A privacy result of no
leakage describes the training regime rather than the defense.

`differential_cifar10_3way.yaml` now trains to a comparable accuracy, so its parity check
compares three backends that have actually learned rather than three sitting at chance.

## 0.4.2

`docs/assets` holds only assets now. The brand kit's README, brand guide, mkdocs snippet,
and preview page were build-time material rather than anything the site or the package
uses, and none of them was referenced.

## 0.4.1

**Reporting.** The aggregation rule is now an explicit field. `RunSpec.aggregation()` names
`fedavg` or the robust rule that replaced it, it appears in each run's recorded parameters,
and the run matrix gives it a column or lists it among the shared settings. It was
previously only inferable from the defense list.

Every shortened column is explained in a legend printed under the table, so `asr`,
`pc-min`, and `mia-auc` no longer send a reader to the source. A per-round trace of the
headline metric sits alongside it, since the table alone showed where a run ended but not
how it got there.

## 0.4.0

**Membership inference.** Added the `membership_inference` attack, which asks whether a
record was in a client's training data. It is the canonical privacy attack the proposal
cites and the one Pitfall-1 says evaluations skip. An honest-but-curious server scores the
global model each round by per-sample loss, following Yeom et al., since a model assigns
lower loss to what it trained on. Members are the target client's data and non-members are
the held-out test set.

It records `membership_inference_auc`, where 0.5 is no leakage, and `membership_loss_gap`.
No shadow model is needed, and because it reads only losses it works on text as well as
images, which gradient inversion does not.

`examples/configs/membership_inference.yaml` runs one overfitted setup twice. Undefended it
reaches AUC 0.67 with a loss gap of 1.39. Clipping with Gaussian noise takes the AUC to
0.50, which is chance, and the gap to 0.01, for about six points of accuracy. That is the
privacy and utility trade-off of Pitfall-4, measured.

The pitfall checker counts it as a privacy attack, so `P5_subtle_leakage` now recommends it
ahead of gradient inversion, which is cheaper to run and applies to more datasets.

## 0.3.4

**Fixed.** The NVFlare backend only ever worked on 1-channel, 10-class data. It rebuilds the
model in its server process from the class path and recovers constructor arguments by
reading attributes of the same name off the instance. The built-in models did not expose
`channels` or `num_classes`, so NVFlare fell back to their defaults and sent every client a
model shaped for MNIST. CIFAR-10 failed on that path before this release, and so did
CIFAR-100 and FEMNIST. The models now expose those arguments, and
`examples/configs/differential_cifar10_3way.yaml` is a three-way parity check on 3-channel
data, which the MNIST example could not catch.

**Fixed.** NVFlare round snapshots are keyed by round number in a cache that was never
cleared between runs. A run that produced no snapshots of its own replayed the previous
run's and reported them as its results. The cache is now cleared alongside the workspace.

**Changed.** The NVFlare backend now refuses a torchvision or Hugging Face model with a
message naming the built-in models and the backends that do run it. It previously failed
inside NVFlare with a JSON encoding error.

## 0.3.3

Documentation uses the light scheme only, so the dark toggle and its maroon page background
are gone. Removed the brand section from the landing page.

## 0.3.2

**CI.** Added `.github/workflows/ci.yml`, which runs on every pull request and on pushes to
the default branch. One job installs from a clean checkout and then runs `fltest list`,
checking the catalog rather than the exit code alone. That is the job that would have
caught the packaging bug where `fltest/data` existed locally but was never committed. A
second job runs the test suite, and a third builds the documentation with `--strict`, so a
broken link or a missing asset fails the build.

**Docs.** The worked example claimed its report file was present. Reports are generated
rather than checked in, so it now says what running the config writes.

## 0.3.1

**Branding.** The documentation site now carries the FLTest identity. `brand.css` is loaded
as `extra_css`, and it binds the maroon, orange, and stone tokens to Material's variables
for both the light and dark schemes. The palette is declared as `custom` so those tokens
govern the colours, since naming a built-in Material palette would fight them.

The header uses the white and orange mark, which is the variant drawn to read on maroon,
and the favicon comes from the same set. The home page and the README show the full lockup
and swap it by colour scheme, through `#only-light` and `#only-dark` on the site and a
`<picture>` element on GitHub. The brand kit itself is linked from the home page.

## 0.3.0

**Text.** Added `ag_news` and the plumbing federated text needs. A dataset now declares its
modality, text splits are tokenized rather than transformed, and a model named `hf:<id>` on
a text dataset is built as a sequence classifier. One function, `forward_batch`, is the only
place that knows an image batch carries `img` while a text batch carries `input_ids` and
`attention_mask`, so the training and evaluation loops serve both.

A `tokenizer` knob was added. It defaults to the Hugging Face model's own tokenizer and is
set explicitly when a repository ships no fast tokenizer but shares another model's
vocabulary. Token ids are part of the dataset cache key.

`examples/configs/text_domains.yaml` gives each client a single news topic, which is the
extreme non-IID setting for language data, and runs an IID baseline beside it. The IID run
reaches 0.4629 accuracy with a worst client at 0.4617, while one topic per client falls to
0.2402 against a 1/4 chance baseline with a worst client at 0.0000.

**Guards.** `backdoor` stamps a trigger onto pixels and `dlg` reconstructs pixels from
gradients, so neither applies to text. Both now refuse a text run with a message naming the
attacks that do apply, rather than failing on a missing column.

## 0.2.0

**Datasets.** Added `cifar100` and `femnist`. FEMNIST is the answer to Pitfall-2, because it
labels every character by the writer who produced it. The new `natural` partitioner gives
each client one real writer instead of a synthetic shard. A dataset name FLTest does
not recognise is now treated as a Hugging Face id and described from its metadata, so any
Hub image-classification dataset works without a code change. A dataset that ships no test
split, as FEMNIST does, gets 10,000 examples held out under a fixed seed before
partitioning. Slicing the test set out of the client shards instead would evaluate the
global model on data its own clients trained on.

**Models.** Added the torchvision architectures `ResNet18`, `ResNet34`, `ResNet50`,
`VGG11`, `MobileNetV3`, and `EfficientNetB0`, each adapted to the dataset's channel count,
with the ResNet stem replaced by the 3x3 CIFAR variant. A name written as `hf:<id>` is
fetched from the Hugging Face Hub through timm or transformers, which installs with
`pip install -e ".[hf]"`.

**Pitfall checker.** CIFAR-100 joins the class-balanced set and FEMNIST is deliberately
outside it, so `dataset: femnist` now clears `P2_dataset` rather than downgrading it. The
recommender suggests FEMNIST, which previously proposed a counter-experiment that could not
clear the pitfall it was answering.

**Fixed.** The deterministic initial-weight cache was keyed on model name and channel count
alone. Adding CIFAR-100 exposed it: a 10-class head would have been loaded into a 100-class
model. The key now carries the class count.

## 0.1.0

First versioned release. It covers Tasks 1 to 3 of the project plan, which are automated
test orchestration, FL input configuration, and evaluation metrics with reporting.

**Orchestration.** A single YAML config expands into a grid of runs through the config
fuzzer, and every run executes behind one `run_simulation()` adapter.

**Backends.** A dependency-light reference oracle, Flower, and NVFlare as an optional
extra. Backends are declared lazily, so `fltest list` and `fltest pitfalls` return without
importing torch.

**Attacks.** `label_flip`, `sign_flip`, `gaussian`, `backdoor` with attack success rate,
and `dlg` gradient inversion with reconstruction MSE, PSNR, and label recovery.

**Defenses.** `gradient_noise`, `norm_clip`, and robust aggregation by `krum`,
`trimmed_mean`, and `median`.

**Testing.** Cross-framework differential parity, a determinism mode, four metamorphic
relations, and a pitfall checker that emits counter-experiments.

**Reporting.** An aligned run-matrix table that states shared parameters once and gives a
column to every parameter that differs, alongside a JSON report for each command.
