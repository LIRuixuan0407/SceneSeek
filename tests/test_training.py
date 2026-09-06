from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sceneseek.training.data import (
    TemporalFeatureDataset,
    _sample_video_timestamps,
    collate_temporal_batch,
)
from sceneseek.training.evaluate import (
    evaluate_mean_pooling,
    retrieval_metrics_from_embeddings,
)
from sceneseek.training.losses import (
    hard_negative_margin_loss,
    multi_positive_contrastive_loss,
)
from sceneseek.training.temporal_adapter import (
    TemporalAdapterConfig,
    create_temporal_adapter,
)

torch = pytest.importorskip("torch")


def _make_feature_store(root: Path) -> Path:
    (root / "videos").mkdir(parents=True)
    np.savez_compressed(
        root / "videos" / "a.npz",
        frames=np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        timestamps=np.asarray([0.0, 1.0], dtype=np.float32),
    )
    np.savez_compressed(
        root / "videos" / "b.npz",
        frames=np.asarray([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32),
        timestamps=np.asarray([0.0, 1.0], dtype=np.float32),
    )
    np.save(
        root / "texts.npy",
        np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
    )
    rows = [
        {
            "sample_id": "a-1",
            "video_id": "a",
            "caption": "red",
            "split": "train",
            "video_feature_file": "videos/a.npz",
            "text_index": 0,
        },
        {
            "sample_id": "b-1",
            "video_id": "b",
            "caption": "blue",
            "split": "train",
            "video_feature_file": "videos/b.npz",
            "text_index": 1,
        },
        {
            "sample_id": "a-2",
            "video_id": "a",
            "caption": "scarlet",
            "split": "test",
            "video_feature_file": "videos/a.npz",
            "text_index": 2,
        },
    ]
    with (root / "manifest.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    return root


def test_temporal_adapter_masks_padding_and_normalizes() -> None:
    config = TemporalAdapterConfig(
        dimension=8,
        num_layers=1,
        num_heads=2,
        dropout=0.0,
        max_frames=4,
    )
    model = create_temporal_adapter(config).eval()
    frames = torch.randn(2, 4, 8)
    mask = torch.tensor([[True, True, False, False], [True, True, True, True]])
    with torch.inference_mode():
        output = model(frames, mask)
    assert output.shape == (2, 8)
    assert torch.linalg.vector_norm(output, dim=-1).tolist() == pytest.approx([1.0, 1.0])


def test_multi_positive_loss_does_not_treat_same_video_as_negative() -> None:
    videos = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    texts = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    good = multi_positive_contrastive_loss(videos, texts, ["a", "a", "b"], temperature=0.1)
    bad = multi_positive_contrastive_loss(videos, texts, ["a1", "a2", "b"], temperature=0.1)
    assert float(good) < float(bad)


def test_hard_negative_margin_loss_rewards_positive_gap() -> None:
    texts = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    positives = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    easy_negatives = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    hard_negatives = torch.tensor([[0.99, 0.01], [0.01, 0.99]])

    easy_loss = hard_negative_margin_loss(texts, positives, easy_negatives, margin=0.05)
    hard_loss = hard_negative_margin_loss(texts, positives, hard_negatives, margin=0.05)

    assert float(easy_loss) == pytest.approx(0.0)
    assert float(hard_loss) > float(easy_loss)


def test_temporal_feature_dataset_collates_variable_length(tmp_path: Path) -> None:
    root = _make_feature_store(tmp_path / "features")
    dataset = TemporalFeatureDataset(root, split="train")
    batch = collate_temporal_batch([dataset[0], dataset[1]])
    assert batch["frames"].shape == (2, 2, 2)
    assert batch["mask"].all()
    assert batch["video_ids"] == ["a", "b"]


def test_retrieval_metrics_and_mean_pool_baseline(tmp_path: Path) -> None:
    metrics = retrieval_metrics_from_embeddings(
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        ["a", "b"],
        np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        ["a", "b"],
    )
    assert metrics["r@1"] == 1.0
    assert metrics["mrr"] == 1.0

    root = _make_feature_store(tmp_path / "features")
    baseline = evaluate_mean_pooling(root, split="test")
    assert baseline["r@1"] == 1.0


def test_trained_checkpoint_can_be_loaded_and_evaluated(tmp_path: Path) -> None:
    from sceneseek.training.evaluate import evaluate_temporal_checkpoint

    root = _make_feature_store(tmp_path / "features")
    # Re-label one sample as validation so the training pipeline has a real held-out split.
    rows = [json.loads(line) for line in (root / "manifest.jsonl").read_text().splitlines()]
    rows[1]["split"] = "val"
    with (root / "manifest.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")

    config = TemporalAdapterConfig(
        dimension=2,
        num_layers=1,
        num_heads=1,
        dropout=0.0,
        max_frames=2,
    )
    model = create_temporal_adapter(config)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"config": config.to_dict(), "state_dict": model.state_dict()}, checkpoint)
    metrics = evaluate_temporal_checkpoint(
        root,
        checkpoint,
        split="test",
        device="cpu",
        batch_size=2,
    )
    assert metrics["queries"] == 1.0
    assert metrics["videos"] == 1.0


def test_temporal_sampling_uses_bin_centers_away_from_video_tail() -> None:
    duration = 11.67
    timestamps = _sample_video_timestamps(duration, sample_fps=1.0, max_frames=16)

    assert len(timestamps) == 12
    step = duration / len(timestamps)
    assert timestamps[0] == pytest.approx(step / 2.0, abs=0.001)
    assert timestamps[-1] == pytest.approx(duration - step / 2.0, abs=0.001)
    assert timestamps[-1] < duration - 0.05
