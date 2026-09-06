from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sceneseek.training.reranker import RerankerConfig, create_query_conditioned_reranker
from sceneseek.training.reranker_data import RerankerCandidateStore
from sceneseek.training.reranker_train import _metrics_from_ranks

torch = pytest.importorskip("torch")


def _make_store(root: Path) -> Path:
    (root / "videos").mkdir(parents=True)
    videos = {
        "a": np.asarray([[1.0, 0.0], [0.8, 0.2]], dtype=np.float32),
        "b": np.asarray([[0.7, 0.3], [0.6, 0.4]], dtype=np.float32),
    }
    for video_id, frames in videos.items():
        np.savez_compressed(
            root / "videos" / f"{video_id}.npz",
            frames=frames,
            timestamps=np.asarray([0.0, 1.0], dtype=np.float32),
        )
    np.save(root / "texts.npy", np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    rows = [
        {
            "sample_id": "a-1",
            "video_id": "a",
            "caption": "a",
            "split": "train",
            "video_feature_file": "videos/a.npz",
            "text_index": 0,
        },
        {
            "sample_id": "b-1",
            "video_id": "b",
            "caption": "b",
            "split": "train",
            "video_feature_file": "videos/b.npz",
            "text_index": 1,
        },
    ]
    with (root / "manifest.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    return root


def test_query_conditioned_reranker_starts_from_stage1_scores() -> None:
    config = RerankerConfig(
        dimension=8,
        num_heads=2,
        hidden_dimension=16,
        dropout=0.0,
        max_frames=4,
    )
    model = create_query_conditioned_reranker(config).eval()
    texts = torch.randn(3, 8)
    frames = torch.randn(3, 4, 8)
    mask = torch.ones(3, 4, dtype=torch.bool)
    coarse = torch.tensor([0.1, 0.3, 0.2])
    with torch.inference_mode():
        scores = model(texts, frames, mask, coarse)
    assert scores.tolist() == pytest.approx(coarse.tolist())


def test_reranker_candidate_store_validates_and_loads_candidates(tmp_path: Path) -> None:
    root = _make_store(tmp_path / "features")
    candidate_path = tmp_path / "candidates.npz"
    np.savez_compressed(
        candidate_path,
        candidates=np.asarray([[0, 1], [1, 0]], dtype=np.int32),
        scores=np.asarray([[0.9, 0.8], [0.7, 0.6]], dtype=np.float32),
        target_ranks=np.asarray([1, 1], dtype=np.int32),
        target_positions=np.asarray([0, 0], dtype=np.int32),
        target_columns=np.asarray([0, 1], dtype=np.int32),
        sample_ids=np.asarray(["a-1", "b-1"]),
        video_ids=np.asarray(["a", "b"]),
    )
    store = RerankerCandidateStore(root, candidate_path, split="train")
    assert store.top_k == 2
    assert len(store.candidate_video_items(0)) == 2
    assert store.text(0).tolist() == pytest.approx([1.0, 0.0])


def test_reranked_metrics_preserve_full_rank_semantics() -> None:
    metrics = _metrics_from_ranks(np.asarray([1, 3, 12], dtype=np.int32), videos=100)
    assert metrics["r@1"] == pytest.approx(1 / 3)
    assert metrics["r@5"] == pytest.approx(2 / 3)
    assert metrics["r@10"] == pytest.approx(2 / 3)
    assert metrics["median_rank"] == 3.0
