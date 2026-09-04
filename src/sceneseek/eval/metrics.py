from __future__ import annotations

import math
from collections.abc import Sequence


def recall_at_k(ranked_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    retrieved = set(ranked_ids[: max(k, 0)])
    return len(retrieved & relevant_ids) / len(relevant_ids)


def reciprocal_rank(ranked_ids: Sequence[str], relevant_ids: set[str]) -> float:
    for rank, item_id in enumerate(ranked_ids, start=1):
        if item_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(relevances: Sequence[float], k: int) -> float:
    selected = list(relevances[: max(k, 0)])
    dcg = sum((2**value - 1) / math.log2(rank + 1) for rank, value in enumerate(selected, 1))
    ideal = sorted(relevances, reverse=True)[: max(k, 0)]
    idcg = sum((2**value - 1) / math.log2(rank + 1) for rank, value in enumerate(ideal, 1))
    return dcg / idcg if idcg else 0.0


def temporal_iou(predicted: tuple[float, float], target: tuple[float, float]) -> float:
    pred_start, pred_end = predicted
    target_start, target_end = target
    intersection = max(0.0, min(pred_end, target_end) - max(pred_start, target_start))
    union = max(pred_end, target_end) - min(pred_start, target_start)
    return intersection / union if union > 0 else 0.0
