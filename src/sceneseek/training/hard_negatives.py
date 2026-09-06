from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from sceneseek.training.data import TemporalFeatureDataset


def _normalize(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.clip(norms, 1e-12, None)


def mine_hard_negative_index(
    feature_dir: Path,
    output: Path | None = None,
    *,
    split: str = "train",
    top_k: int = 16,
    batch_size: int = 2048,
    device: str = "auto",
) -> dict[str, object]:
    """Mine globally hard negative videos with the frozen CLIP feature store.

    Each caption is compared against the mean-pooled representation of every video in the
    requested split. The ground-truth video is masked, then the top-k nearest wrong videos
    are persisted as compact integer indices. The result can be reused across training runs.
    """
    if top_k <= 0:
        raise ValueError("top_k 必须大于 0")
    if batch_size <= 0:
        raise ValueError("batch_size 必须大于 0")

    dataset = TemporalFeatureDataset(feature_dir, split=split)
    records_by_video: dict[str, int] = {}
    for index, record in enumerate(dataset.records):
        records_by_video.setdefault(record.video_id, index)
    video_ids = list(records_by_video)
    if len(video_ids) < 2:
        raise ValueError("hard negative mining 至少需要两个不同视频")

    effective_k = min(top_k, len(video_ids) - 1)
    video_vectors = []
    for video_id in video_ids:
        item = dataset[records_by_video[video_id]]
        frames = np.asarray(item["frames"], dtype=np.float32)
        video_vectors.append(_normalize(frames.mean(axis=0, keepdims=True))[0])
    videos = np.stack(video_vectors).astype(np.float32, copy=False)
    video_lookup = {video_id: index for index, video_id in enumerate(video_ids)}
    target_columns = np.asarray(
        [video_lookup[record.video_id] for record in dataset.records], dtype=np.int64
    )
    text_indices = np.asarray([record.text_index for record in dataset.records], dtype=np.int64)

    candidates = np.empty((len(dataset), effective_k), dtype=np.int32)
    backend = "numpy"
    try:
        import torch
    except ImportError:
        torch = None

    if device == "auto":
        device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"

    if torch is not None and device != "cpu":
        backend = device
        video_tensor = torch.from_numpy(videos).to(device)
        for start in range(0, len(dataset), batch_size):
            end = min(start + batch_size, len(dataset))
            text_array = np.asarray(dataset.texts[text_indices[start:end]], dtype=np.float32)
            text_array = _normalize(text_array)
            text_tensor = torch.from_numpy(text_array).to(device)
            scores = text_tensor @ video_tensor.T
            rows = torch.arange(end - start, device=device)
            targets = torch.from_numpy(target_columns[start:end]).to(device)
            scores[rows, targets] = float("-inf")
            indices = torch.topk(scores, k=effective_k, dim=1, largest=True, sorted=True).indices
            candidates[start:end] = indices.cpu().numpy().astype(np.int32, copy=False)
    else:
        for start in range(0, len(dataset), batch_size):
            end = min(start + batch_size, len(dataset))
            text_array = np.asarray(dataset.texts[text_indices[start:end]], dtype=np.float32)
            scores = _normalize(text_array) @ videos.T
            scores[np.arange(end - start), target_columns[start:end]] = -np.inf
            partition = np.argpartition(-scores, kth=effective_k - 1, axis=1)[:, :effective_k]
            partition_scores = np.take_along_axis(scores, partition, axis=1)
            order = np.argsort(-partition_scores, axis=1)
            candidates[start:end] = np.take_along_axis(partition, order, axis=1).astype(
                np.int32, copy=False
            )

    output = output or (feature_dir / f"hard-negatives-{split}.npz")
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "split": split,
        "top_k": effective_k,
        "samples": len(dataset),
        "videos": len(video_ids),
        "backend": backend,
    }
    np.savez_compressed(
        output,
        candidates=candidates,
        sample_ids=np.asarray([record.sample_id for record in dataset.records]),
        video_ids=np.asarray(video_ids),
        metadata=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )
    return {**metadata, "output": str(output.resolve())}


class HardNegativeBatchSampler:
    """Build batches whose second half contains mined negatives for the first half.

    Batch size stays constant, so hard-negative training does not double per-epoch forward
    work. Every yielded item is still a normal positive caption/video pair; the existing
    multi-positive loss therefore remains valid while anchors are guaranteed to see close
    non-matching videos in the same batch.
    """

    def __init__(
        self,
        dataset: TemporalFeatureDataset,
        index_path: Path,
        *,
        batch_size: int,
        candidate_pool: int = 16,
        seed: int = 7,
    ) -> None:
        if batch_size < 2:
            raise ValueError("hard negative batch_size 至少为 2")
        if candidate_pool <= 0:
            raise ValueError("candidate_pool 必须大于 0")
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0

        with np.load(index_path, allow_pickle=False) as payload:
            candidates = np.asarray(payload["candidates"], dtype=np.int32)
            sample_ids = [str(value) for value in payload["sample_ids"].tolist()]
            video_ids = [str(value) for value in payload["video_ids"].tolist()]
        expected = [record.sample_id for record in dataset.records]
        if sample_ids != expected:
            raise ValueError("hard negative index 与当前 feature split/sample 顺序不匹配")
        if candidates.shape[0] != len(dataset) or candidates.ndim != 2:
            raise ValueError("hard negative index candidates shape 无效")
        self.candidates = candidates[:, : min(candidate_pool, candidates.shape[1])]
        self.video_ids = video_ids

        records_by_video: dict[str, list[int]] = defaultdict(list)
        for index, record in enumerate(dataset.records):
            records_by_video[record.video_id].append(index)
        missing = [video_id for video_id in video_ids if video_id not in records_by_video]
        if missing:
            raise ValueError(f"hard negative index 包含当前 split 不存在的视频: {missing[0]}")
        self.records_by_video = records_by_video

    def __len__(self) -> int:
        return math.ceil(len(self.dataset) / self.batch_size)

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        shuffled = list(range(len(self.dataset)))
        rng.shuffle(shuffled)

        for start in range(0, len(shuffled), self.batch_size):
            source = shuffled[start : start + self.batch_size]
            if len(source) < 2:
                yield source
                continue
            anchor_count = max(1, (len(source) + 1) // 2)
            anchors = source[:anchor_count]
            batch = list(anchors)
            needed = len(source) - len(batch)
            for anchor_index in anchors[:needed]:
                candidate_columns = self.candidates[anchor_index]
                column = int(candidate_columns[rng.randrange(len(candidate_columns))])
                video_id = self.video_ids[column]
                batch.append(rng.choice(self.records_by_video[video_id]))
            rng.shuffle(batch)
            yield batch


class HardNegativeLookup:
    """Resolve one mined hard-negative video for each positive training sample."""

    def __init__(
        self,
        dataset: TemporalFeatureDataset,
        index_path: Path,
        *,
        candidate_pool: int = 16,
        seed: int = 7,
    ) -> None:
        if candidate_pool <= 0:
            raise ValueError("candidate_pool 必须大于 0")
        self.dataset = dataset
        self.seed = seed

        with np.load(index_path, allow_pickle=False) as payload:
            candidates = np.asarray(payload["candidates"], dtype=np.int32)
            sample_ids = [str(value) for value in payload["sample_ids"].tolist()]
            video_ids = [str(value) for value in payload["video_ids"].tolist()]

        expected = [record.sample_id for record in dataset.records]
        if sample_ids != expected:
            raise ValueError("hard negative index 与当前 feature split/sample 顺序不匹配")
        if candidates.shape[0] != len(dataset) or candidates.ndim != 2:
            raise ValueError("hard negative index candidates shape 无效")

        self.candidates = candidates[:, : min(candidate_pool, candidates.shape[1])]
        self.video_ids = video_ids
        self.sample_rows = {sample_id: index for index, sample_id in enumerate(sample_ids)}

        first_record_by_video: dict[str, int] = {}
        for index, record in enumerate(dataset.records):
            first_record_by_video.setdefault(record.video_id, index)
        missing = [video_id for video_id in video_ids if video_id not in first_record_by_video]
        if missing:
            raise ValueError(f"hard negative index 包含当前 split 不存在的视频: {missing[0]}")
        self.first_record_by_video = first_record_by_video

    def sample_dataset_indices(
        self,
        sample_ids: list[str],
        *,
        epoch: int,
        step: int,
    ) -> list[int]:
        rng = random.Random(self.seed + epoch * 1_000_003 + step * 97)
        indices: list[int] = []
        for sample_id in sample_ids:
            try:
                row = self.sample_rows[sample_id]
            except KeyError as error:
                raise ValueError(f"hard negative index 缺少 sample_id={sample_id}") from error
            candidate_columns = self.candidates[row]
            column = int(candidate_columns[rng.randrange(len(candidate_columns))])
            video_id = self.video_ids[column]
            indices.append(self.first_record_by_video[video_id])
        return indices
