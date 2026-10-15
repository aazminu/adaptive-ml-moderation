from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple


@dataclass(frozen=True)
class ModerationEventRecord:
    event_id: str
    thread_id: str
    author_id: str
    timestamp_epoch: float
    text: str
    label: int


def split_temporal_thread_disjoint(
    records: Sequence[ModerationEventRecord],
    train_fraction: float = 0.7,
) -> Tuple[List[ModerationEventRecord], List[ModerationEventRecord]]:
    """Splits moderation records by thread start time while enforcing zero thread_id overlap."""
    if not (0.1 <= train_fraction <= 0.9):
        raise ValueError("train_fraction must be in [0.1, 0.9]")
    if len(records) < 2:
        raise ValueError("At least 2 records are required to split")

    thread_earliest: dict[str, float] = {}
    for r in records:
        if r.thread_id not in thread_earliest or r.timestamp_epoch < thread_earliest[r.thread_id]:
            thread_earliest[r.thread_id] = r.timestamp_epoch

    sorted_threads = sorted(thread_earliest.items(), key=lambda kv: (kv[1], kv[0]))
    if len(sorted_threads) < 2:
        raise ValueError("At least 2 distinct thread_ids are required for thread-disjoint splitting")

    cut_idx = max(1, min(len(sorted_threads) - 1, int(round(len(sorted_threads) * train_fraction))))
    train_threads = {t for t, _ in sorted_threads[:cut_idx]}

    train_set = [r for r in records if r.thread_id in train_threads]
    eval_set = [r for r in records if r.thread_id not in train_threads]

    # Verify strict thread disjointness
    overlap = {r.thread_id for r in train_set} & {r.thread_id for r in eval_set}
    if overlap:
        raise RuntimeError(f"Thread leakage detected: {overlap}")
    return train_set, eval_set


def compute_population_stability_index(
    baseline_scores: Sequence[float],
    current_scores: Sequence[float],
    num_bins: int = 5,
) -> float:
    """Computes Population Stability Index (PSI) across [0, 1] risk probability buckets."""
    if not baseline_scores or not current_scores:
        raise ValueError("baseline_scores and current_scores must be non-empty")
    if num_bins < 2:
        raise ValueError("num_bins must be >= 2")

    eps = 1e-4
    base_counts = [0] * num_bins
    curr_counts = [0] * num_bins

    for s in baseline_scores:
        idx = min(num_bins - 1, max(0, int(s * num_bins)))
        base_counts[idx] += 1
    for s in current_scores:
        idx = min(num_bins - 1, max(0, int(s * num_bins)))
        curr_counts[idx] += 1

    psi = 0.0
    n_base = len(baseline_scores)
    n_curr = len(current_scores)
    for b_cnt, c_cnt in zip(base_counts, curr_counts):
        p = max(b_cnt / n_base, eps)
        q = max(c_cnt / n_curr, eps)
        psi += (q - p) * math.log(q / p)
    return round(psi, 5)
