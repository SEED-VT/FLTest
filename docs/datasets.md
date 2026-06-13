# Datasets

FLTest loads and partitions datasets with [`flwr-datasets`](https://flower.ai/docs/datasets/)
(backed by Hugging Face). Code: `fltest/data/datasets.py`.

## Built-in datasets

| Name | Channels | Classes | Image column | Notes |
|------|:--------:|:-------:|--------------|-------|
| `mnist` | 1 | 10 | `image` | handwritten digits |
| `fashion_mnist` | 1 | 10 | `image` | clothing; harder than MNIST, same shape |
| `cifar10` | 3 | 10 | `img` | natural images (RGB) |

Use one with `dataset: cifar10`, or fuzz several with `dataset: [mnist, fashion_mnist, cifar10]`.
Channels and class count are derived automatically — you never set them by hand.

Grayscale datasets are resized to 32×32 and normalized to mean/std 0.5; RGB datasets are
normalized per-channel to 0.5. (Defined in `_TRANSFORMS`.)

## Partitioning (data distribution)

| `data_distribution` | Effect | Relevant knob |
|---------------------|--------|---------------|
| `iid` | uniform random split; every client sees all classes | — |
| `dirichlet` | label skew across clients | `dirichlet_alpha` (lower ⇒ more skewed) |
| `pathological` | each client gets only N classes | `classes_per_partition` |

Non-IID partitioning is how you stress robustness/privacy realistically (the proposal's
Pitfall-2/3). Example:

```yaml
dataset: cifar10
data_distribution: dirichlet
dirichlet_alpha: 0.1        # strongly non-IID
num_clients: 10
```

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
