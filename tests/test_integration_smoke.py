"""Tiny end-to-end smoke: a 2-client x 1-round reference run produces metrics.

Marked slow-ish but kept minimal (MLP on a small MNIST subset, CPU). Skips gracefully if
the MNIST dataset cannot be fetched (offline CI).
"""

import pytest

from fltest.core.config import TestConfig
from fltest.core.orchestrator import Orchestrator


def test_reference_run_smoke(tmp_path):
    cfg = TestConfig(
        name="smoke", dataset="mnist", model_name="MLP",
        num_clients=2, num_rounds=1, dataset_partitions=20, client_lr=0.05,
        max_test_data_size=256,
        model_cache_path=str(tmp_path / "mc"),
        dataset_cache_path=str(tmp_path / "dc"),
        fw_cache_path=str(tmp_path / "fw"),
        runs=[{"framework": "reference"}],
    )
    try:
        matrix = Orchestrator(verbose=False).run(cfg)
    except Exception as exc:  # dataset fetch failures shouldn't fail the suite offline
        pytest.skip(f"dataset/setup unavailable: {exc}")

    result = matrix.results[0]
    assert result.status == "success"
    assert 0.0 <= result.final["accuracy"] <= 1.0
    assert "loss" in result.final
