"""Federated dataset loading & partitioning (built on flwr-datasets).

Supports IID and three non-IID partitioners, which are Dirichlet label skew,
pathological N-classes-per-client, and a natural partition on a real-world client column
such as the FEMNIST writer id. Heterogeneity is what proposal Pitfall-2 (overlooking
dataset sensitivities) and Pitfall-3 (IID-only evaluation) are about.

A dataset that is not named in :data:`DATASET_CONFIG` is treated as a Hugging Face id and
described by inspecting its metadata, so any Hub image-classification dataset is usable
without editing this file.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
from diskcache import Index
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import (
    DirichletPartitioner,
    IidPartitioner,
    NaturalIdPartitioner,
    PathologicalPartitioner,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from fltest.data.utils import seed_everything


@dataclass(frozen=True)
class DatasetSpec:
    """Everything FLTest needs to know about a dataset to federate it."""

    hf_id: str                      # id passed to flwr-datasets / Hugging Face
    column: str                     # column holding the input (an image, or text)
    modality: str = "image"         # "image" or "text"
    label_column: str = "label"     # column holding the class label
    channels: int = 1               # 1 grayscale, 3 RGB
    num_classes: int = 10
    transform: str = "grayscale"    # key into _TRANSFORMS
    natural_partition_by: str = ""  # column giving a real-world client id, if the data has one
    test_split: str = "test"        # split to evaluate on; "" means hold one out of train
    holdout_size: int = 10_000      # examples held out when test_split is ""
    max_length: int = 128           # tokens kept per example, text only


DATASET_CONFIG: Dict[str, DatasetSpec] = {
    # Hub ids are always `namespace/name`. huggingface-hub 1.16 dropped the bare form, so
    # `mnist` raises HfUriError there unless a stale local cache happens to answer for it.
    "mnist": DatasetSpec("ylecun/mnist", "image", channels=1, num_classes=10, transform="grayscale"),
    "fashion_mnist": DatasetSpec(
        "zalando-datasets/fashion_mnist", "image", channels=1, num_classes=10,
        transform="grayscale",
    ),
    "cifar10": DatasetSpec("uoft-cs/cifar10", "img", channels=3, num_classes=10, transform="rgb"),
    "cifar100": DatasetSpec(
        "uoft-cs/cifar100", "img", label_column="fine_label",
        channels=3, num_classes=100, transform="rgb",
    ),
    # FEMNIST is the naturally non-IID dataset the proposal calls out: handwritten
    # characters labelled by writer, so `data_distribution: natural` gives each client one
    # real writer instead of a synthetic shard.
    "femnist": DatasetSpec(
        "flwrlabs/femnist", "image", label_column="character",
        channels=1, num_classes=62, transform="grayscale",
        natural_partition_by="writer_id",
        # FEMNIST ships a single train split of 814k examples, so a central test set has to
        # be held out before partitioning. Slicing it out of the client shards instead would
        # evaluate the global model on data its own clients trained on.
        test_split="",
    ),
    # AG News topics stand in for domains. Partitioning it with `pathological` and
    # `classes_per_partition: 1` gives each client a single topic, which is the
    # one-domain-per-client setting used to study non-IID federated language models.
    "ag_news": DatasetSpec(
        "fancyzhx/ag_news", "text", modality="text",
        channels=0, num_classes=4, transform="",
    ),
}

_TRANSFORMS = {
    "rgb": transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    ),
    "grayscale": transforms.Compose(
        [transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    ),
}

# name -> factory(num_partitions, label_column, **kwargs) -> Partitioner.
# A lower Dirichlet alpha means more label skew across clients.
PARTITIONERS = {
    "iid": lambda n, label_column="label", **kw: IidPartitioner(num_partitions=n),
    "dirichlet": lambda n, label_column="label", alpha=0.5, seed=42, **kw: DirichletPartitioner(
        num_partitions=n, partition_by=label_column, alpha=alpha, seed=seed
    ),
    "pathological": lambda n, label_column="label", classes_per_partition=2, seed=42, **kw: (
        PathologicalPartitioner(
            num_partitions=n, partition_by=label_column,
            num_classes_per_partition=classes_per_partition, seed=seed,
        )
    ),
    "natural": lambda n, label_column="label", partition_by="", **kw: NaturalIdPartitioner(
        partition_by=partition_by
    ),
}

#: Fixed seed for the train/test holdout, so a dataset without a test split still gives
#: the same evaluation set on every run and across every framework.
_HOLDOUT_SEED = 786

_probe_cache: Dict[str, DatasetSpec] = {}


def _probe_hub(dataset_name: str) -> DatasetSpec:
    """Describe an unlisted dataset by reading its Hub metadata.

    Only the dataset card and feature schema are fetched, not the data itself, so this is
    cheap. The image column is the first image-valued feature and the label column is the
    first one carrying class names.
    """
    if dataset_name in _probe_cache:
        return _probe_cache[dataset_name]
    from datasets import load_dataset_builder

    features = load_dataset_builder(dataset_name).info.features
    image_col = next(
        (k for k, v in features.items() if type(v).__name__ == "Image"), None
    )
    label_col = next((k for k, v in features.items() if hasattr(v, "names")), None)
    if image_col is None or label_col is None:
        raise ValueError(
            f"Cannot use '{dataset_name}' automatically: FLTest needs one image column and "
            f"one labelled class column, but found {list(features)}. Add an entry to "
            f"DATASET_CONFIG in fltest/data/datasets.py to describe it explicitly."
        )
    num_classes = len(features[label_col].names)
    spec = DatasetSpec(
        dataset_name, image_col, label_column=label_col,
        channels=3, num_classes=num_classes, transform="rgb",
    )
    _probe_cache[dataset_name] = spec
    return spec


def resolve_dataset(dataset_name: str) -> DatasetSpec:
    """Return the :class:`DatasetSpec` for a built-in name or a Hugging Face id."""
    if dataset_name in DATASET_CONFIG:
        return DATASET_CONFIG[dataset_name]
    if "/" not in dataset_name:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Built-in names are {list_datasets()}. Any "
            f"other dataset is a Hugging Face id and must be written as 'namespace/name', "
            f"for example 'uoft-cs/cifar10'. The bare form was removed in "
            f"huggingface-hub 1.16."
        )
    return _probe_hub(dataset_name)


def list_datasets() -> List[str]:
    return sorted(DATASET_CONFIG)


def list_partitioners() -> List[str]:
    return sorted(PARTITIONERS)


def dataset_meta(dataset_name: str):
    """Return (channels, num_classes) for a built-in dataset or a Hugging Face id."""
    spec = resolve_dataset(dataset_name)
    return spec.channels, spec.num_classes


def _tokenize_split(split, spec: DatasetSpec, tokenizer_id: str):
    """Tokenise a text split into ``input_ids`` / ``attention_mask`` / ``label``."""
    if not tokenizer_id:
        raise ValueError(
            f"'{spec.hf_id}' is a text dataset, so FLTest needs a tokenizer. Set a Hugging "
            f"Face model with `model_name: hf:<id>` and its tokenizer is used, or name one "
            f"explicitly with `tokenizer: <id>`."
        )
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            'Text datasets need the Hugging Face extra. Install it with pip install -e ".[hf]"'
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

    def encode(batch):
        return tokenizer(
            batch, padding="max_length", truncation=True, max_length=spec.max_length
        )

    keep = {"input_ids", "attention_mask", "label"}
    split = split.map(encode, input_columns=spec.column, batched=True)
    return split.remove_columns([c for c in split.column_names if c not in keep])


def get_federated_dataset(
    dataset_name: str,
    num_clients: int,
    partitioner: str = "iid",
    tokenizer_id: str = "",
    **part_kwargs,
):
    """Partition ``dataset_name`` into ``num_clients`` shards (HF datasets, not loaders)."""
    if partitioner not in PARTITIONERS:
        raise ValueError(f"Unknown partitioner '{partitioner}'. Available: {list_partitioners()}")

    spec = resolve_dataset(dataset_name)

    if partitioner == "natural":
        if not spec.natural_partition_by:
            raise ValueError(
                f"'{dataset_name}' has no natural client column, so data_distribution "
                f"'natural' does not apply. Datasets that do: "
                f"{[n for n, s in DATASET_CONFIG.items() if s.natural_partition_by]}."
            )
        part_kwargs = {**part_kwargs, "partition_by": spec.natural_partition_by}

    part = PARTITIONERS[partitioner](num_clients, label_column=spec.label_column, **part_kwargs)

    if spec.test_split:
        fds = FederatedDataset(dataset=spec.hf_id, partitioners={"train": part})
        test_raw = fds.load_split(spec.test_split)
        client_raw = {cid: fds.load_partition(cid) for cid in range(num_clients)}
    else:
        from datasets import load_dataset

        full = load_dataset(spec.hf_id, split="train")
        holdout = full.train_test_split(
            test_size=min(spec.holdout_size, len(full) // 10), seed=_HOLDOUT_SEED, shuffle=True
        )
        part.dataset = holdout["train"]  # partition only the training portion
        test_raw = holdout["test"]
        client_raw = {cid: part.load_partition(cid) for cid in range(num_clients)}

    def prepare(split):
        # Every downstream consumer reads batch["label"], so normalise the label column
        # name here rather than teaching the training loops about each dataset.
        if spec.label_column != "label":
            split = split.rename_column(spec.label_column, "label")
        if spec.modality == "text":
            split = _tokenize_split(split, spec, tokenizer_id)
        else:
            transform = _TRANSFORMS[spec.transform]
            split = split.map(lambda img: {"img": transform(img)}, input_columns=spec.column)
        return split.with_format("torch")

    return {
        "c2data": {cid: prepare(raw) for cid, raw in client_raw.items()},
        "test_data": prepare(test_raw),
    }


def get_cached_federated_dataset(
    dataset_name: str,
    num_clients: int,
    cache_path: str,
    partitioner: str = "iid",
    tokenizer_id: str = "",
    **part_kwargs,
):
    """Cached wrapper around :func:`get_federated_dataset`.

    The key covers the tokenizer too, since two models with different tokenizers produce
    different token ids from the same text.
    """
    cache = Index(cache_path)
    kw = "_".join(f"{k}{v}" for k, v in sorted(part_kwargs.items()))
    key = f"{dataset_name}_{num_clients}_{partitioner}_{tokenizer_id}_{kw}"
    if key not in cache:
        cache[key] = get_federated_dataset(
            dataset_name, num_clients, partitioner, tokenizer_id=tokenizer_id, **part_kwargs
        )
    return cache[key]


class _InMemoryShard(Dataset):
    """A prepared split's columns, converted to tensors once.

    ``prepare`` stores transformed images as nested lists, and the torch format rebuilds
    them element by element on every read. Across a run that is most of the wall-clock
    time, and a GPU cannot help with it. The conversion here goes through the same torch
    formatter, just once for the whole split, so every column comes out with the dtype and
    shape a batch read gives, and batches, shuffling and results are unchanged.
    """

    def __init__(self, columns: Dict[str, torch.Tensor]):
        self.columns = columns
        self._length = len(next(iter(columns.values())))

    @classmethod
    def from_split(cls, split):
        """The split in memory, or None when a column is ragged and cannot be stacked."""
        if len(split) == 0:
            return None
        columns = split.with_format("torch")[:]
        if not columns or not all(isinstance(v, torch.Tensor) for v in columns.values()):
            return None
        return cls(dict(columns))

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        return {name: column[index] for name, column in self.columns.items()}


def _in_memory(split):
    """Serve ``split`` from memory when it stacks into tensors, else read it as before."""
    return _InMemoryShard.from_split(split) or split


def build_dataloaders(
    dataset_dict: Dict,
    num_clients: int,
    client_batch_size: int,
    server_batch_size: int,
    max_test_size: int,
    seed: int,
):
    """Wrap HF dataset shards in torch DataLoaders (used by reference + Flower backends).

    Only the first ``num_clients`` shards are returned as loaders (the dataset may be
    partitioned more finely than the number of participating clients).
    """
    seed_everything(seed)

    def worker_init_fn(worker_id):
        np.random.seed(seed + worker_id)

    c2loader = {
        cid: DataLoader(
            _in_memory(dataset_dict["c2data"][cid]),
            batch_size=client_batch_size,
            shuffle=True,
            num_workers=0,
            worker_init_fn=worker_init_fn,
        )
        for cid in range(num_clients)
    }
    test_split = dataset_dict["test_data"]
    n_test = min(max_test_size, len(test_split))
    test_loader = DataLoader(
        _in_memory(test_split.select(range(n_test))),
        batch_size=server_batch_size,
        shuffle=False,
        num_workers=0,
    )
    return {"c2loader": c2loader, "test_loader": test_loader}


def client_label_counts(dataset_dict: Dict) -> Dict[int, Dict[int, int]]:
    """Per-client class histogram — used by the pitfall checker to detect IID-only setups."""
    counts: Dict[int, Dict[int, int]] = {}
    for cid, data in sorted(dataset_dict["c2data"].items()):
        dset = data.dataset if hasattr(data, "dataset") else data
        labels = dset["label"]
        if torch.is_tensor(labels):
            labels = labels.tolist()
        counts[cid] = dict(Counter(int(x) for x in labels))
    return counts
