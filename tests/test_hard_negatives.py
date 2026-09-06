from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sceneseek.training.data import TemporalFeatureDataset
from sceneseek.training.hard_negatives import (
    HardNegativeBatchSampler,
    HardNegativeLookup,
    mine_hard_negative_index,
)

pytest.importorskip("torch")


def _make_store(root: Path) -> Path:
    (root / "videos").mkdir(parents=True)
    vectors = {
        "a": np.asarray([1.0, 0.0], dtype=np.float32),
        "b": np.asarray([0.9, 0.1], dtype=np.float32),
        "c": np.asarray([0.0, 1.0], dtype=np.float32),
        "d": np.asarray([-1.0, 0.0], dtype=np.float32),
    }
    rows = []
    texts = []
    for text_index, (video_id, vector) in enumerate(vectors.items()):
        np.savez_compressed(
            root / "videos" / f"{video_id}.npz",
            frames=np.stack([vector, vector]),
            timestamps=np.asarray([0.0, 1.0], dtype=np.float32),
        )
        texts.append(vector)
        rows.append(
            {
                "sample_id": f"sample-{video_id}",
                "video_id": video_id,
                "caption": video_id,
                "split": "train",
                "video_feature_file": f"videos/{video_id}.npz",
                "text_index": text_index,
            }
        )
    np.save(root / "texts.npy", np.stack(texts))
    with (root / "manifest.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    return root


def test_global_hard_negative_mining_excludes_ground_truth(tmp_path: Path) -> None:
    root = _make_store(tmp_path / "features")
    index_path = tmp_path / "hard-negatives.npz"
    result = mine_hard_negative_index(
        root,
        index_path,
        split="train",
        top_k=2,
        batch_size=2,
        device="cpu",
    )
    assert result["samples"] == 4
    with np.load(index_path, allow_pickle=False) as payload:
        video_ids = [str(value) for value in payload["video_ids"].tolist()]
        candidates = np.asarray(payload["candidates"])
    assert video_ids[int(candidates[0, 0])] == "b"
    for row, video_id in enumerate(video_ids):
        assert video_id not in {video_ids[int(column)] for column in candidates[row]}


def test_hard_negative_sampler_injects_mined_partner(tmp_path: Path) -> None:
    root = _make_store(tmp_path / "features")
    index_path = tmp_path / "hard-negatives.npz"
    mine_hard_negative_index(root, index_path, top_k=1, device="cpu")
    dataset = TemporalFeatureDataset(root, split="train")
    sampler = HardNegativeBatchSampler(
        dataset,
        index_path,
        batch_size=4,
        candidate_pool=1,
        seed=3,
    )
    batch = next(iter(sampler))
    assert len(batch) == 4
    batch_video_ids = [dataset.records[index].video_id for index in batch]
    # At least one nearest-neighbour pair (a,b) must be present in the mined batch.
    assert "a" in batch_video_ids and "b" in batch_video_ids


def test_hard_negative_lookup_resolves_wrong_video(tmp_path: Path) -> None:
    root = _make_store(tmp_path / "features")
    index_path = tmp_path / "hard-negatives.npz"
    mine_hard_negative_index(root, index_path, top_k=2, device="cpu")
    dataset = TemporalFeatureDataset(root, split="train")
    lookup = HardNegativeLookup(
        dataset,
        index_path,
        candidate_pool=1,
        seed=5,
    )

    sample_ids = [record.sample_id for record in dataset.records]
    negative_indices = lookup.sample_dataset_indices(sample_ids, epoch=1, step=0)
    for positive, negative_index in zip(dataset.records, negative_indices, strict=True):
        assert dataset.records[negative_index].video_id != positive.video_id
