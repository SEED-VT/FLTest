"""Online FLDetector: history-based detection before aggregation.

This keeps the paper's prediction, score window, and gap-statistic decision, but
filters a detected client immediately and in later rounds instead of restarting
training from the initial model. FLTest submissions contain model weights, so the
gradient-like vector used here is ``global_weights - client_weights``.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from fltest.core.hook_context import HookContext
from fltest.core.registry import register_defense
from fltest.defenses.base import PPFLBaseClass


def _flatten(arrays) -> np.ndarray:
    return np.concatenate([np.asarray(a, dtype=np.float64).ravel() for a in arrays])


def _hessian_product(pairs, vector: np.ndarray) -> np.ndarray:
    """Apply a limited-memory BFGS Hessian (not its inverse) to a vector."""
    usable = [
        (s, y) for s, y in pairs
        if np.dot(s, s) > 1e-12 and np.dot(y, s) > 1e-12
        and np.isfinite(s).all() and np.isfinite(y).all()
    ]
    if not usable:
        return np.zeros_like(vector)
    s_last, y_last = usable[-1]
    scale = float(np.dot(y_last, s_last) / np.dot(s_last, s_last))
    terms = []

    def apply(value):
        out = scale * value
        for bs, sbs, y, ys in terms:
            out = out - bs * (np.dot(bs, value) / sbs) + y * (np.dot(y, value) / ys)
        return out

    for s, y in usable:
        ys = float(np.dot(y, s))
        if ys <= 1e-12 or not np.isfinite(ys):
            continue
        bs = apply(s)
        sbs = float(np.dot(s, bs))
        if sbs <= 1e-12 or not np.isfinite(sbs):
            continue
        terms.append((bs, sbs, y, ys))
    return apply(vector)


def _cluster_costs(values: np.ndarray, max_clusters: int):
    """Exact one-dimensional k-means costs and split points."""
    values = np.sort(np.asarray(values, dtype=np.float64))
    n = len(values)
    sums = np.r_[0.0, np.cumsum(values)]
    squares = np.r_[0.0, np.cumsum(values * values)]
    cost = np.full((max_clusters + 1, n + 1), np.inf)
    split = np.zeros((max_clusters + 1, n + 1), dtype=int)
    cost[0, 0] = 0.0
    for k in range(1, max_clusters + 1):
        for end in range(k, n + 1):
            starts = np.arange(k - 1, end)
            count = end - starts
            total = sums[end] - sums[starts]
            sse = squares[end] - squares[starts] - total * total / count
            candidates = cost[k - 1, starts] + np.maximum(sse, 0.0)
            best = int(np.argmin(candidates))
            cost[k, end] = candidates[best]
            split[k, end] = starts[best]
    return cost[:, n], split


def _suspected_ids(scores: dict[int, float], max_clusters: int, samples: int, seed: int):
    if len(scores) < 3:
        return set()
    ids = sorted(scores)
    values = np.array([scores[cid] for cid in ids], dtype=np.float64)
    spread = float(np.ptp(values))
    if not np.isfinite(spread) or spread <= 1e-12:
        return set()
    values = (values - values.min()) / spread
    limit = min(max_clusters, len(values) - 1)
    observed, _ = _cluster_costs(values, limit + 1)
    rng = np.random.default_rng(seed)
    references = np.array([
        _cluster_costs(rng.uniform(size=len(values)), limit + 1)[0][1:]
        for _ in range(samples)
    ])
    eps = np.finfo(float).tiny
    ref_logs = np.log(np.maximum(references, eps))
    gaps = ref_logs.mean(axis=0) - np.log(np.maximum(observed[1:], eps))
    deviation = np.sqrt(1.0 + 1.0 / samples) * ref_logs.std(axis=0, ddof=1)
    chosen = next(
        (k for k in range(1, limit + 1)
         if gaps[k - 1] >= gaps[k] - deviation[k]),
        limit + 1,
    )
    if chosen == 1:
        return set()
    _, two_split = _cluster_costs(values, 2)
    boundary = two_split[2, len(values)]
    sorted_ids = sorted(ids, key=lambda cid: scores[cid])
    return set(sorted_ids[boundary:])


@register_defense("fldetector")
class FLDetectorDefense(PPFLBaseClass):
    """Detect clients with repeatedly inconsistent updates, then exclude them.

    Must precede an optional robust aggregator in the defense list. Requires
    identified submissions from the reference or Flower backend.
    """

    HOOKS = ("before_simulation", "before_round", "before_aggregate", "after_aggregate")

    def __init__(self, window_size=10, start_round=50, max_clusters=10, gap_samples=20, **params):
        super().__init__(**params)
        if window_size < 1 or start_round < 1 or max_clusters < 2 or gap_samples < 2:
            raise ValueError("Invalid FLDetector window, start round, cluster, or sample count")
        self.window_size = int(window_size)
        self.start_round = int(start_round)
        self.max_clusters = int(max_clusters)
        self.gap_samples = int(gap_samples)
        self._reset()

    def _reset(self):
        self.excluded_clients: set[int] = set()
        self._previous_updates: dict[int, np.ndarray] = {}
        self._previous_model: np.ndarray | None = None
        self._previous_gradient: np.ndarray | None = None
        self._pairs = deque(maxlen=self.window_size)
        self._distances = deque(maxlen=self.window_size)
        self._current_model: np.ndarray | None = None

    def before_simulation(self, ctx: HookContext) -> None:
        self._reset()

    def before_round(self, ctx: HookContext) -> None:
        if not self.excluded_clients:
            return
        selected = ctx.selected_clients
        if selected is None:
            selected = range(ctx.cfg.num_clients)
        ctx.selected_clients = tuple(cid for cid in selected if cid not in self.excluded_clients)
        if not ctx.selected_clients:
            raise ValueError("FLDetector excluded every available client")

    def before_aggregate(self, ctx: HookContext) -> None:
        submissions = ctx.client_submissions
        uw = ctx.updates_and_weights
        if not submissions or uw is None or ctx.global_state is None:
            raise ValueError("FLDetector requires identified submissions and a global model")
        if len(submissions) != len(uw) or any(
            len(record.update) != len(update)
            or any(a is not b for a, b in zip(record.update, update))
            for record, (update, _) in zip(submissions, uw)
        ):
            raise ValueError("FLDetector requires submissions aligned with aggregation inputs")
        ids = [record.client_id for record in submissions]
        if any(cid is None for cid in ids) or len(set(ids)) != len(ids):
            raise ValueError("FLDetector requires unique, stable client IDs")

        model = _flatten(ctx.global_state)
        updates = {record.client_id: model - _flatten(record.update) for record in submissions}
        distances = {}
        if self._previous_model is not None and self._pairs:
            hvp = _hessian_product(self._pairs, model - self._previous_model)
            distances = {
                cid: float(np.linalg.norm(update - self._previous_updates[cid] - hvp))
                for cid, update in updates.items() if cid in self._previous_updates
            }
            total = sum(distances.values())
            if total > 0:
                distances = {cid: value / total for cid, value in distances.items()}
            else:
                distances = {cid: 0.0 for cid in distances}
            self._distances.append(distances)

        if (not self.excluded_clients and ctx.round is not None
                and ctx.round >= self.start_round and len(self._pairs) >= self.window_size
                and len(self._distances) >= self.window_size):
            common = set(updates).intersection(*(set(row) for row in self._distances))
            scores = {
                cid: float(np.mean([row[cid] for row in self._distances]))
                for cid in common
            }
            detected = _suspected_ids(
                scores, self.max_clusters, self.gap_samples,
                int(ctx.cfg.seed) + int(ctx.round),
            )
            self.excluded_clients.update(detected)
            ctx.record(
                fldetector_scores=scores,
                fldetector_detected_clients=sorted(detected),
                fldetector_detected_count=len(detected),
            )

        self._previous_updates = {cid: value.copy() for cid, value in updates.items()}
        self._current_model = model.copy()
        if self.excluded_clients:
            ctx.updates_and_weights = [
                pair for cid, pair in zip(ids, uw) if cid not in self.excluded_clients
            ]
            if not ctx.updates_and_weights:
                raise ValueError("FLDetector filtered every received update")

    def after_aggregate(self, ctx: HookContext) -> None:
        current = self._current_model
        if current is None or ctx.new_global_state is None:
            return
        gradient = current - _flatten(ctx.new_global_state)
        if self._previous_model is not None and self._previous_gradient is not None:
            self._pairs.append((
                current - self._previous_model,
                gradient - self._previous_gradient,
            ))
        self._previous_model = current.copy()
        self._previous_gradient = gradient.copy()
