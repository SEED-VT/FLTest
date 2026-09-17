"""Validate a before_round hook's requested client participation."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple


def resolve_selected_clients(
    selected: Optional[Sequence[int]], num_clients: int
) -> Tuple[int, ...]:
    """Return ordered, nonempty partition IDs; ``None`` retains all clients."""
    if selected is None:
        selected = tuple(range(num_clients))
    if not isinstance(selected, (list, tuple)) or not selected:
        raise ValueError("before_round selected_clients must be a nonempty list or tuple")
    if any(
        not isinstance(cid, int) or isinstance(cid, bool) or not 0 <= cid < num_clients
        for cid in selected
    ):
        raise ValueError(
            f"before_round selected_clients must contain client IDs from 0 to {num_clients - 1}"
        )
    if len(set(selected)) != len(selected):
        raise ValueError("before_round selected_clients must not contain duplicates")
    return tuple(selected)
