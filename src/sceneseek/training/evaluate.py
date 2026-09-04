from __future__ import annotations

import math
from collections import OrderedDict
from pathlib import Path

import numpy as np

from sceneseek.training.data import TemporalFeatureDataset


def _normalize(array: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(array, axis=-1, keepdims=True)
    return array / np.clip(norms, 1e-12, None)


def retrieval_metrics_from_embeddings(
    text_embeddings: np.ndarray,
    text_video_ids: list[str],
    video_embeddings: np.ndarray,
    video_ids: list[str],
    *,
    chunk_size: int = 1024,
) -> dict[str, float]:
    texts = _normalize(np.asarray(text_embeddings, dtype=np.float32))
    videos = _normalize(np.asarray(video_embeddings, dtype=np.float32))
    if len(text_video_ids) != texts.shape[0] or len(video_ids) != videos.shape[0]:
        raise ValueError("embedding 数量与 id 数量不一致")
    lookup = {video_id: index for index, video_id in enumerate(video_ids)}
    ranks: list[int] = []
    for start in range(0, texts.shape[0], max(1, chunk_size)):
        similarities = texts[start : start + chunk_size] @ videos.T
        for offset, scores in enumerate(similarities):
            target_id = text_video_ids[start + offset]
            if target_id not in lookup:
                raise ValueError(f"query 对应的视频不存在: {target_id}")
            target = lookup[target_id]
            target_score = float(scores[target])
            # Optimistic tie handling is deterministic and matches exact-ranking baselines.
            rank = 1 + int(np.sum(scores > target_score))
            ranks.append(rank)
    values = np.asarray(ranks, dtype=np.int64)
    return {
        "queries": float(len(ranks)),
        "videos": float(len(video_ids)),
        "r@1": float(np.mean(values <= 1)),
        "r@5": float(np.mean(values <= 5)),
        "r@10": float(np.mean(values <= 10)),
        "mrr": float(np.mean(1.0 / values)),
        "median_rank": float(np.median(values)),
        "mean_rank": float(np.mean(values)),
    }


def _unique_video_records(dataset: TemporalFeatureDataset):
    unique: OrderedDict[str, int] = OrderedDict()
    for index, record in enumerate(dataset.records):
        unique.setdefault(record.video_id, index)
    return unique


def evaluate_mean_pooling(feature_dir: Path, *, split: str = "test") -> dict[str, float]:
    dataset = TemporalFeatureDataset(feature_dir, split=split)
    unique = _unique_video_records(dataset)
    video_ids: list[str] = []
    video_vectors: list[np.ndarray] = []
    for video_id, index in unique.items():
        item = dataset[index]
        frames = np.asarray(item["frames"], dtype=np.float32)
        video_ids.append(video_id)
        video_vectors.append(_normalize(frames.mean(axis=0, keepdims=True))[0])
    texts = np.stack([np.asarray(dataset[index]["text"], dtype=np.float32) for index in range(len(dataset))])
    text_video_ids = [record.video_id for record in dataset.records]
    return retrieval_metrics_from_embeddings(
        texts,
        text_video_ids,
        np.stack(video_vectors),
        video_ids,
    )


def load_temporal_checkpoint(path: Path, device: str = "cpu"):
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("请先安装 `pip install -e '.[ml]'`") from error

    from sceneseek.training.temporal_adapter import TemporalAdapterConfig, create_temporal_adapter

    payload = torch.load(path, map_location=device, weights_only=False)
    config = TemporalAdapterConfig(**payload["config"])
    model = create_temporal_adapter(config)
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    return model, payload


def evaluate_temporal_checkpoint(
    feature_dir: Path,
    checkpoint: Path,
    *,
    split: str = "test",
    device: str = "auto",
    batch_size: int = 64,
) -> dict[str, float]:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("请先安装 `pip install -e '.[ml]'`") from error

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_temporal_checkpoint(checkpoint, device)
    dataset = TemporalFeatureDataset(feature_dir, split=split)
    unique = _unique_video_records(dataset)
    video_ids = list(unique.keys())
    video_vectors: list[np.ndarray] = []

    entries = list(unique.items())
    for start in range(0, len(entries), max(1, batch_size)):
        batch = entries[start : start + batch_size]
        arrays = [np.asarray(dataset[index]["frames"], dtype=np.float32) for _, index in batch]
        max_steps = max(array.shape[0] for array in arrays)
        dimension = arrays[0].shape[1]
        frames = torch.zeros((len(arrays), max_steps, dimension), dtype=torch.float32, device=device)
        mask = torch.zeros((len(arrays), max_steps), dtype=torch.bool, device=device)
        for row, array in enumerate(arrays):
            steps = array.shape[0]
            frames[row, :steps] = torch.from_numpy(array).to(device)
            mask[row, :steps] = True
        with torch.inference_mode():
            encoded = model(frames, mask).cpu().numpy()
        video_vectors.extend(encoded)

    texts = np.stack([np.asarray(dataset[index]["text"], dtype=np.float32) for index in range(len(dataset))])
    text_video_ids = [record.video_id for record in dataset.records]
    return retrieval_metrics_from_embeddings(
        texts,
        text_video_ids,
        np.stack(video_vectors),
        video_ids,
    )
