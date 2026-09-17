# Datasets

FLTest loads and partitions datasets with [`flwr-datasets`](https://flower.ai/docs/datasets/)
(backed by Hugging Face). Code: `fltest/data/datasets.py`.

## Built-in datasets

| Name | Channels | Classes | Notes |
|------|:--------:|:-------:|-------|
| `mnist` | 1 | 10 | handwritten digits |
| `fashion_mnist` | 1 | 10 | clothing; harder than MNIST, same shape |
| `cifar10` | 3 | 10 | natural images (RGB) |
| `cifar100` | 3 | 100 | 100 fine-grained classes; labels live in `fine_label` |
| `femnist` | 1 | 62 | handwritten characters **labelled by writer**; naturally non-IID |
| `ag_news` | text | 4 | news topics; needs the `[hf]` extra (see below) |

Use one with `dataset: cifar10`, or fuzz several with `dataset: [mnist, cifar100, femnist]`.
Channels and class count are derived automatically, so you never set them by hand.

### Any Hugging Face dataset

A name FLTest does not recognise is treated as a Hub id. Its metadata is read to find the
image column and the label column, and the class count follows from the label feature.

```yaml
dataset: uoft-cs/cifar100
```

A Hub id must be written as `namespace/name`. huggingface-hub 1.16 removed the bare form,
so `cifar10` on its own raises `HfUriError` on a machine that has not cached it already.
FLTest says so rather than letting the Hub error through. The built-in short names are
unaffected, since each maps to a namespaced id internally.

Only the dataset card and feature schema are fetched for this, not the data. A dataset
without exactly one image column and one labelled class column raises an error naming the
columns it did find, and the fix is an explicit entry in `DATASET_CONFIG`.

### FEMNIST and the balanced-dataset pitfall

MNIST, Fashion-MNIST, CIFAR-10, and CIFAR-100 are class balanced, so the pitfall checker
flags a configuration that uses only those (`P2_dataset`). FEMNIST is the way out. Each
character carries the id of the writer who produced it, so `data_distribution: natural`
gives every client one real person's handwriting rather than a synthetic shard.

```bash
fltest run examples/configs/femnist_natural.yaml
```

That example trains 40 writers for 10 rounds and reports 0.2656 global accuracy against a
1/62 chance baseline. The per-client numbers are the point: mean accuracy is 0.2623 while
the worst-served writer sits at 0.0599, which is the representation disparity a single
global number hides.

FEMNIST publishes only a train split, so FLTest holds out 10,000 examples with a fixed seed
before partitioning. Carving the test set out of the client shards instead would evaluate
the global model on data its own clients trained on.

Grayscale datasets are resized to 32×32 and normalized to mean/std 0.5; RGB datasets are
normalized per-channel to 0.5. (Defined in `_TRANSFORMS`.)

## Partitioning (data distribution)

| `data_distribution` | Effect | Relevant knob |
|---------------------|--------|---------------|
| `iid` | uniform random split; every client sees all classes | — |
| `dirichlet` | label skew across clients | `dirichlet_alpha` (lower ⇒ more skewed) |
| `pathological` | each client gets only N classes | `classes_per_partition` |
| `natural` | one client per real-world id in the data | dataset must define one |

`natural` works only on a dataset that carries a client column, which today means FEMNIST
and its `writer_id`. Asking for it elsewhere raises an error naming the datasets that
support it.

Non-IID partitioning is how you stress robustness/privacy realistically (the project's
Pitfall-2/3). Example:

```yaml
dataset: cifar10
data_distribution: dirichlet
dirichlet_alpha: 0.1        # strongly non-IID
num_clients: 10
```

## Text datasets

`ag_news` is a text dataset, so it needs a tokenizer and a Hugging Face sequence
classifier. Install the extra with `pip install -e ".[hf]"`.

```yaml
dataset: ag_news
model_name: hf:google/bert_uncased_L-2_H-128_A-2
tokenizer: google-bert/bert-base-uncased   # optional, see below
```

The tokenizer defaults to the model's own. Name one explicitly when a model repository
ships no fast tokenizer but shares another model's vocabulary. That is the case above,
where both use the same 30522-token WordPiece vocabulary. Token ids are
part of the dataset cache key, so two models with different tokenizers never share a cache
entry.

### One domain per client

AG News labels each article as world, sports, business, or sci-tech. Partitioning it with
`pathological` and one class per client gives every client a single topic, so no two
clients share a domain. That is the text analogue of the single-class-per-client split used
to stress CNNs.

```bash
fltest run examples/configs/text_domains.yaml
```

That example runs the same setup twice, changing only how topics are spread:

| run | distribution | accuracy | worst client |
|-----|--------------|:--------:|:------------:|
| `iid` | iid | 0.4629 | 0.4617 |
| `one_topic_per_client` | pathological, 1 class | 0.2402 | **0.0000** |

With one topic per client the global model falls to chance for four classes. At least one
client scores nothing at all. The IID run reaches 0.4629 from identical data, model, and
budget.

### What does not apply to text

`backdoor` stamps a trigger patch onto pixels and `dlg` reconstructs pixels from
gradients, so neither applies to a text run. Both raise an error saying so rather than
failing on a missing column. `label_flip`, `sign_flip`, and `gaussian` work on labels and
updates, so they apply to either modality, as do every defense and metric.

## Use an existing dataset

Just name it:

```yaml
dataset: fashion_mnist
data_distribution: pathological
classes_per_partition: 2
```

## Attach a new dataset

Two small edits in `fltest/data/datasets.py`:

**1. Register it in `DATASET_CONFIG`** with `(transform_key, image_column, channels, classes)`:

```python
DATASET_CONFIG = {
    "mnist": ("grayscale", "image", 1, 10),
    "cifar10": ("rgb", "img", 3, 10),
    # new: a 3-channel, 100-class HF dataset whose image column is "img"
    "cifar100": ("rgb", "img", 3, 100),
}
```

- `transform_key` selects a transform in `_TRANSFORMS` (`"grayscale"` or `"rgb"`). Add a new
  key there if your data needs a different transform.
- `image_column` is the Hugging Face column holding the image (often `image` or `img`).
- `channels` and `classes` are surfaced to the model and metrics.

**2. (Only if needed) add a transform** in `_TRANSFORMS`, e.g. for 28×28 inputs without
resizing or for different normalization.

That's it — `dataset: cifar100` now works, including fuzzing and all partitioners. The HF
dataset name passed to `flwr-datasets` is the key you used (`"cifar100"`); use the
fully-qualified HF id (e.g. `"zalando-datasets/fashion_mnist"`) if the short name is
ambiguous.

### Custom / local data

`get_federated_dataset()` returns `{"c2data": {cid: hf_dataset}, "test_data": hf_dataset}`
where each shard yields `{"img": tensor, "label": tensor}` after transform. To plug in data
that isn't on Hugging Face, build those dicts yourself (any object exposing
`{"img","label"}` batches works) and call `build_dataloaders(...)`, or add a new partitioner
to `PARTITIONERS`.

## Sweeping several at once

`examples/configs/datasets_showcase.yaml` crosses three datasets with two distributions,
which is six runs from one file:

```bash
fltest run      examples/configs/datasets_showcase.yaml
fltest pitfalls examples/configs/datasets_showcase.yaml
```

It is also a useful input to the pitfall checker, because sweeping more than one dataset
and more than one distribution clears the MNIST-only and IID-only findings that a
single-dataset config raises.

## Add a new partitioner

`PARTITIONERS` maps a name to a factory `f(num_partitions, **kwargs) -> Partitioner`:

```python
PARTITIONERS = {
    "iid": lambda n, **kw: IidPartitioner(num_partitions=n),
    "my_skew": lambda n, alpha=0.3, **kw: DirichletPartitioner(
        num_partitions=n, partition_by="label", alpha=alpha),
}
```

Then `data_distribution: my_skew` is usable from any config.
