"""Dataset specs and the model zoo, covering the paths that need no network."""

import pytest
import torch

from fltest.data.datasets import (
    DATASET_CONFIG,
    build_dataloaders,
    dataset_meta,
    get_federated_dataset,
    list_partitioners,
    resolve_dataset,
)
from fltest.data.models import TORCHVISION_MODELS, get_model, list_models


def test_dataset_specs_describe_their_label_columns():
    """A dataset whose labels are not called 'label' must say so."""
    assert resolve_dataset("cifar100").label_column == "fine_label"
    assert resolve_dataset("femnist").label_column == "character"
    assert resolve_dataset("mnist").label_column == "label"


def test_dataset_meta_covers_the_new_datasets():
    assert dataset_meta("cifar100") == (3, 100)
    assert dataset_meta("femnist") == (1, 62)


def test_femnist_is_the_only_naturally_partitioned_dataset():
    """FEMNIST carries a writer id, which is what `data_distribution: natural` needs."""
    natural = {n for n, s in DATASET_CONFIG.items() if s.natural_partition_by}
    assert natural == {"femnist"}
    assert resolve_dataset("femnist").natural_partition_by == "writer_id"
    assert "natural" in list_partitioners()


def test_femnist_holds_out_its_own_test_set():
    """FEMNIST ships only a train split, so a held-out set has to be carved from it."""
    assert resolve_dataset("femnist").test_split == ""
    assert resolve_dataset("cifar100").test_split == "test"


def test_natural_partitioning_is_refused_where_it_does_not_apply():
    with pytest.raises(ValueError, match="no natural client column"):
        get_federated_dataset("mnist", 2, "natural")


@pytest.mark.parametrize("name", ["LeNet", "MLP", "ResNet18", "MobileNetV3"])
def test_models_accept_the_dataset_shape(name):
    """Every model adapts to the channel count and class count it is given."""
    model = get_model(name, "data/models_cache", channels=1, num_classes=62)
    assert model(torch.zeros(2, 1, 32, 32)).shape == (2, 62)


def test_torchvision_architectures_are_listed():
    assert set(TORCHVISION_MODELS) <= set(list_models())
    assert {"LeNet", "ConvNet", "MLP"} <= set(list_models())


def test_initial_weight_cache_separates_class_counts():
    """The cache key carries num_classes, or a 10-class head would be loaded into a 100."""
    ten = get_model("LeNet", "data/models_cache", channels=3, num_classes=10)
    hundred = get_model("LeNet", "data/models_cache", channels=3, num_classes=100)
    assert ten.fc3.out_features == 10
    assert hundred.fc3.out_features == 100


def test_unknown_model_name_explains_the_options():
    with pytest.raises(ValueError, match="hf:"):
        get_model("NoSuchNet", "data/models_cache", channels=1, num_classes=10)


def test_every_dataset_id_is_namespaced():
    """huggingface-hub 1.16 dropped bare repository ids, so `mnist` no longer resolves.

    A machine with a warm cache still answers for the bare form, which is why this needs a
    test rather than a run: the failure only appears on a cold cache.
    """
    bare = {name: spec.hf_id for name, spec in DATASET_CONFIG.items() if "/" not in spec.hf_id}
    assert not bare, f"these must be written as namespace/name: {bare}"


def test_a_bare_unknown_dataset_says_what_is_wrong():
    with pytest.raises(ValueError, match="must be written as 'namespace/name'"):
        resolve_dataset("not_a_builtin")


def _prepared_split(n, width=4):
    """A split shaped like ``prepare``'s output: nested-list images and int labels."""
    from datasets import Dataset

    images = torch.arange(n * width * width, dtype=torch.float32).reshape(n, 1, width, width)
    return Dataset.from_dict(
        {"img": images.tolist(), "label": [i % 3 for i in range(n)]}
    ).with_format("torch")


def test_client_loaders_serve_the_same_batches_from_memory():
    from torch.utils.data import DataLoader

    from fltest.data.utils import seed_everything

    shards = {cid: _prepared_split(10 + cid) for cid in range(2)}
    loaders = build_dataloaders(
        {"c2data": shards, "test_data": _prepared_split(7)}, 2, 4, 3, 5, seed=1)
    for cid, shard in shards.items():
        seed_everything(1)
        expected = [b for _ in range(2) for b in DataLoader(shard, batch_size=4, shuffle=True)]
        seed_everything(1)
        served = [b for _ in range(2) for b in loaders["c2loader"][cid]]
        assert len(served) == len(expected)
        for got, want in zip(served, expected):
            assert got.keys() == want.keys()
            for key in want:
                assert got[key].dtype == want[key].dtype
                assert torch.equal(got[key], want[key])
    assert sum(len(b["label"]) for b in loaders["test_loader"]) == 5


def test_a_ragged_split_is_read_as_before():
    from datasets import Dataset

    from fltest.data.datasets import _in_memory

    ragged = Dataset.from_dict({"img": [[1.0], [1.0, 2.0]], "label": [0, 1]}).with_format("torch")
    assert _in_memory(ragged) is ragged
