from __future__ import annotations

import pytest

from sceneseek.eval import ndcg_at_k, recall_at_k, reciprocal_rank, temporal_iou


def test_retrieval_metrics() -> None:
    ranked = ["a", "b", "c", "d"]
    relevant = {"b", "d"}

    assert recall_at_k(ranked, relevant, 2) == 0.5
    assert reciprocal_rank(ranked, relevant) == 0.5
    assert ndcg_at_k([3, 0, 2], 3) == pytest.approx(0.9558, rel=1e-3)


def test_temporal_iou() -> None:
    assert temporal_iou((2, 6), (4, 8)) == pytest.approx(1 / 3)
    assert temporal_iou((0, 1), (2, 3)) == 0
