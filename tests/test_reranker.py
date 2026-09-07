from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sceneseek.training.reranker import RerankerConfig, create_query_conditioned_reranker
from sceneseek.training.reranker_data import RerankerCandidateStore
from sceneseek.training.reranker_train import (
    _metrics_from_ranks,
    evaluate_reranker,
    train_reranker,
)

torch = pytest.importorskip("torch")


def _make_store(root: Path, *, split: str = "train") -> Path:
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
            "split": split,
            "video_feature_file": "videos/a.npz",
            "text_index": 0,
        },
        {
            "sample_id": "b-1",
            "video_id": "b",
            "caption": "b",
            "split": split,
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


def test_query_conditioned_reranker_bounds_residual() -> None:
    config = RerankerConfig(
        dimension=8,
        num_heads=2,
        hidden_dimension=16,
        dropout=0.0,
        max_frames=4,
        residual_scale=0.05,
    )
    model = create_query_conditioned_reranker(config).eval()
    with torch.no_grad():
        model.scorer[-1].bias.fill_(100.0)
    texts = torch.randn(3, 8)
    frames = torch.randn(3, 4, 8)
    mask = torch.ones(3, 4, dtype=torch.bool)
    coarse = torch.tensor([0.1, 0.3, 0.2])
    with torch.inference_mode():
        scores = model(texts, frames, mask, coarse)
    delta = (scores - coarse).abs()
    assert float(delta.max()) <= config.residual_scale + 1e-6


def test_identity_checkpoint_preserves_stage1_and_reports_oracle(tmp_path: Path) -> None:
    root = _make_store(tmp_path / "features", split="val")
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
    config = RerankerConfig(
        dimension=2,
        num_heads=1,
        hidden_dimension=4,
        dropout=0.0,
        max_frames=2,
        residual_scale=0.05,
    )
    model = create_query_conditioned_reranker(config).eval()
    checkpoint = tmp_path / "identity.pt"
    torch.save(
        {
            "format_version": 2,
            "config": config.to_dict(),
            "state_dict": model.state_dict(),
        },
        checkpoint,
    )
    metrics = evaluate_reranker(
        root, candidate_path, checkpoint, split="val", device="cpu", batch_size=2
    )
    for key in ("r@1", "r@5", "r@10", "mrr"):
        assert metrics[key] == pytest.approx(metrics[f"stage1_{key}"])
    assert metrics["oracle_r@1"] == pytest.approx(metrics["candidate_recall"])
    assert metrics["mean_abs_delta"] == pytest.approx(0.0)
    assert metrics["max_abs_delta"] == pytest.approx(0.0)


def test_train_reranker_rejects_injected_positives(tmp_path: Path) -> None:
    root = _make_store(tmp_path / "features")
    train_candidates = tmp_path / "train.npz"
    val_candidates = tmp_path / "val.npz"
    payload = {
        "candidates": np.asarray([[0], [1]], dtype=np.int32),
        "scores": np.asarray([[0.9], [0.8]], dtype=np.float32),
        "target_ranks": np.asarray([2, 1], dtype=np.int32),
        "target_positions": np.asarray([0, 0], dtype=np.int32),
        "target_columns": np.asarray([0, 1], dtype=np.int32),
        "sample_ids": np.asarray(["a-1", "b-1"]),
        "video_ids": np.asarray(["a", "b"]),
    }
    np.savez_compressed(train_candidates, **payload)
    np.savez_compressed(val_candidates, **payload)
    with pytest.raises(ValueError, match="注入"):
        train_reranker(
            root,
            train_candidates,
            val_candidates,
            tmp_path / "artifacts",
            epochs=1,
            batch_size=1,
            num_heads=1,
            hidden_dimension=4,
            max_frames=2,
            device="cpu",
        )
