"""Running Flower clients on a multi-node Ray cluster (FLTEST_RAY_ADDRESS).

Three things must change when the clients leave the driver's machine: Flower joins the existing
cluster instead of starting a private one, the prepared data shards travel by value, because a
memory-mapped Hugging Face table pickles as a path that only exists on the driver, and each
client's shard goes into the object store once, so a message to a client does not carry every
client's data. These checks need no network and no running cluster.
"""

import pytest

from fltest.data import datasets
from fltest.frameworks.flower.adapter import RAY_ADDRESS_ENV, ray_backend_config, share_with_cluster
from fltest.frameworks.flower.client import client_loader


@pytest.fixture
def no_cluster(monkeypatch):
    monkeypatch.delenv(RAY_ADDRESS_ENV, raising=False)


@pytest.fixture
def cluster(monkeypatch):
    monkeypatch.setenv(RAY_ADDRESS_ENV, "10.0.0.1:6379")


def test_local_ray_is_sized_by_the_spec(no_cluster):
    cfg = ray_backend_config(total_cpus=8, total_gpus=2, num_clients=4)
    assert cfg["init_args"] == {"num_cpus": 8, "num_gpus": 2}
    assert cfg["client_resources"] == {"num_cpus": 1, "num_gpus": 0.5}


def test_cluster_address_replaces_local_resources(cluster):
    cfg = ray_backend_config(total_cpus=8, total_gpus=0, num_clients=4)
    # Ray rejects num_cpus/num_gpus when it connects to an existing cluster.
    assert cfg["init_args"] == {"address": "10.0.0.1:6379"}
    assert cfg["client_resources"] == {"num_cpus": 1, "num_gpus": 0.0}


def test_empty_address_means_local(monkeypatch):
    monkeypatch.setenv(RAY_ADDRESS_ENV, "")
    assert "address" not in ray_backend_config(4, 0, 2)["init_args"]


def test_shards_stay_memory_mapped_locally(no_cluster):
    assert datasets._shards_in_memory() is False
    assert datasets._cache_key("mnist", 10, "iid", "", {}) == "mnist_10_iid__"


def test_cluster_shards_are_in_memory_with_their_own_cache_key(cluster):
    assert datasets._shards_in_memory() is True
    key = datasets._cache_key("mnist", 10, "dirichlet", "", {"alpha": 0.5})
    assert key == "mnist_10_dirichlet__alpha0.5_inmem"


def test_local_loaders_are_used_directly():
    loader = [1, 2, 3]
    assert client_loader(loader) is loader


def test_shared_loaders_are_fetched_per_client():
    import ray

    ray.init(num_cpus=1, include_dashboard=False)
    try:
        refs = share_with_cluster({0: [1, 2], 1: [3]}, {"init_args": {}})
        assert all(isinstance(r, ray.ObjectRef) for r in refs.values())
        assert client_loader(refs[1]) == [3]
    finally:
        ray.shutdown()
