"""Parameter <-> model helpers for the Flower backend (numpy-array form)."""

from __future__ import annotations

from collections import OrderedDict
from typing import List

import numpy as np
import torch


def get_parameters(model) -> List[np.ndarray]:
    return [v.detach().cpu().numpy() for v in model.state_dict().values()]


def set_parameters(model, parameters: List[np.ndarray]) -> None:
    keys = list(model.state_dict().keys())
    state = OrderedDict({k: torch.as_tensor(v) for k, v in zip(keys, parameters)})
    model.load_state_dict(state, strict=True)


def numeric_metrics(metrics: dict) -> dict:
    """Keep only JSON/scalar-friendly numeric entries (for Flower fit-metric transport)."""
    out = {}
    for k, v in metrics.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out[k] = float(v)
    return out
