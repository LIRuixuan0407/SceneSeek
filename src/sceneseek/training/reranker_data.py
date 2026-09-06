from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import numpy as np

from sceneseek.training.data import TemporalFeatureDataset
from sceneseek.training.evaluate import load_temporal_checkpoint


def _normalize(array: np.ndarray) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    return values / np.clip(norms, 1e-12, None)


def _unique_video_records(dataset: TemporalFeatureDataset) -> OrderedDict[str, int]:
    unique: OrderedDict[str, int] = OrderedDict()
    for index, record in enumerate(dataset.records):
        unique.setdefault(record.video_id, index)
    return unique


def encode_stage1_videos(
    feature_dir: Path,
    checkpoint: Path,
    *,
    split: str,
    device: str = "auto",
    batch_size: int = 64,
) -> tuple[list[str], np.ndarray]:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("请先安装 `pip install -e '.[ml]'`") from error

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("指定了 CUDA，但当前 PyTorch 无法访问 CUDA")

    model, _ = load_temporal_checkpoint(checkpoint, device)
    dataset = TemporalFeatureDataset(feature_dir, split=split)
    unique = _unique_video_records(dataset)
    entries = list(unique.items())
    video_vectors: list[np.ndarray] = []

    for start in range(0, len(entries), max(1, batch_size)):
        batch = entries[start : start + batch_size]
        arrays = [np.asarray(dataset[index]["frames"], dtype=np.float32) for _, index in batch]
        max_steps = max(array.shape[0] for array in arrays)
        dimension = arrays[0].shape[1]
        frames = torch.zeros(
            (len(arrays), max_steps, dimension), dtype=torch.float32, device=device
        )
        mask = torch.zeros((len(arrays), max_steps), dtype=torch.bool, device=device)
        for row, array in enumerate(arrays):
            steps = array.shape[0]
            frames[row, :steps] = torch.from_numpy(array).to(device)
            mask[row, :steps] = True
        with torch.inference_mode():
            encoded = model(frames, mask).cpu().numpy()
        video_vectors.extend(encoded)

    return list(unique.keys()), _normalize(np.stack(video_vectors))


def build_reranker_candidates(
    feature_dir: Path,
    stage1_checkpoint: Path,
    output: Path,
    *,
    split: str,
    top_k: int = 20,
    device: str = "auto",
    batch_size: int = 64,
    query_batch_size: int = 2048,
    ensure_positive: bool = False,
) -> dict[str, object]:
    """Build Stage-1 candidate sets and preserve exact original target ranks."""
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("请先安装 `pip install -e '.[ml]'`") from error

    if top_k <= 0:
        raise ValueError("top_k 必须大于 0")
    if query_batch_size <= 0:
        raise ValueError("query_batch_size 必须大于 0")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("指定了 CUDA，但当前 PyTorch 无法访问 CUDA")

    dataset = TemporalFeatureDataset(feature_dir, split=split)
    video_ids, video_vectors = encode_stage1_videos(
        feature_dir,
        stage1_checkpoint,
        split=split,
        device=device,
        batch_size=batch_size,
    )
    top_k = min(top_k, len(video_ids))
    video_lookup = {video_id: index for index, video_id in enumerate(video_ids)}
    texts = _normalize(
        np.stack(
            [np.asarray(dataset[index]["text"], dtype=np.float32) for index in range(len(dataset))]
        )
    )
    target_columns = np.asarray(
        [video_lookup[record.video_id] for record in dataset.records], dtype=np.int32
    )

    video_tensor = torch.from_numpy(video_vectors).to(device)
    candidate_columns = np.empty((len(dataset), top_k), dtype=np.int32)
    candidate_scores = np.empty((len(dataset), top_k), dtype=np.float32)
    target_ranks = np.empty(len(dataset), dtype=np.int32)
    target_positions = np.full(len(dataset), -1, dtype=np.int32)

    for start in range(0, len(dataset), query_batch_size):
        end = min(len(dataset), start + query_batch_size)
        text_tensor = torch.from_numpy(texts[start:end]).to(device)
        with torch.inference_mode():
            similarities = text_tensor @ video_tensor.T
            values, columns = torch.topk(similarities, k=top_k, dim=1)
        values_np = values.cpu().numpy().astype(np.float32, copy=False)
        columns_np = columns.cpu().numpy().astype(np.int32, copy=False)
        target_np = target_columns[start:end]
        row_indices = torch.arange(end - start, device=device)
        target_tensor = torch.from_numpy(target_np).to(device)
        target_score = similarities[row_indices, target_tensor]
        ranks = 1 + (similarities > target_score[:, None]).sum(dim=1)
        target_ranks[start:end] = ranks.cpu().numpy().astype(np.int32, copy=False)

        for row in range(end - start):
            matches = np.flatnonzero(columns_np[row] == target_np[row])
            if matches.size:
                target_positions[start + row] = int(matches[0])
            elif ensure_positive:
                columns_np[row, -1] = target_np[row]
                values_np[row, -1] = float(target_score[row].detach().cpu())
                target_positions[start + row] = top_k - 1

        candidate_columns[start:end] = columns_np
        candidate_scores[start:end] = values_np

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        candidates=candidate_columns,
        scores=candidate_scores,
        target_ranks=target_ranks,
        target_positions=target_positions,
        target_columns=target_columns,
        sample_ids=np.asarray([record.sample_id for record in dataset.records]),
        video_ids=np.asarray(video_ids),
        split=np.asarray(split),
        top_k=np.asarray(top_k, dtype=np.int32),
    )
    candidate_recall = float(np.mean(target_ranks <= top_k))
    summary = {
        "split": split,
        "queries": len(dataset),
        "videos": len(video_ids),
        "top_k": top_k,
        "candidate_recall": candidate_recall,
        "ensure_positive": ensure_positive,
        "backend": device,
        "output": str(output.resolve()),
        "stage1_checkpoint": str(stage1_checkpoint.resolve()),
    }
    metadata_path = output.with_suffix(output.suffix + ".json")
    metadata_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


class RerankerCandidateStore:
    def __init__(
        self,
        feature_dir: Path,
        candidate_path: Path,
        *,
        split: str,
        cache_size: int = 512,
    ) -> None:
        self.dataset = TemporalFeatureDataset(feature_dir, split=split, cache_size=cache_size)
        with np.load(candidate_path, allow_pickle=False) as payload:
            self.candidates = np.asarray(payload["candidates"], dtype=np.int32)
            self.scores = np.asarray(payload["scores"], dtype=np.float32)
            self.target_ranks = np.asarray(payload["target_ranks"], dtype=np.int32)
            self.target_positions = np.asarray(payload["target_positions"], dtype=np.int32)
            self.target_columns = np.asarray(payload["target_columns"], dtype=np.int32)
            self.sample_ids = [str(value) for value in payload["sample_ids"].tolist()]
            self.video_ids = [str(value) for value in payload["video_ids"].tolist()]

        expected = [record.sample_id for record in self.dataset.records]
        if self.sample_ids != expected:
            raise ValueError("reranker candidates 与当前 feature split/sample 顺序不匹配")
        if self.candidates.shape != self.scores.shape:
            raise ValueError("candidate columns/scores shape 不一致")
        if self.candidates.shape[0] != len(self.dataset):
            raise ValueError("candidate query 数量与 feature split 不一致")

        first_record_by_video: dict[str, int] = {}
        for index, record in enumerate(self.dataset.records):
            first_record_by_video.setdefault(record.video_id, index)
        missing = [video_id for video_id in self.video_ids if video_id not in first_record_by_video]
        if missing:
            raise ValueError(f"candidate pool 包含当前 split 不存在的视频: {missing[0]}")
        self.video_record_indices = np.asarray(
            [first_record_by_video[video_id] for video_id in self.video_ids], dtype=np.int32
        )

    def __len__(self) -> int:
        return len(self.dataset)

    @property
    def top_k(self) -> int:
        return int(self.candidates.shape[1])

    @property
    def dimension(self) -> int:
        return self.dataset.dimension

    def text(self, query_index: int) -> np.ndarray:
        return np.asarray(self.dataset[query_index]["text"], dtype=np.float32)

    def candidate_video_items(self, query_index: int) -> list[dict[str, object]]:
        columns = self.candidates[query_index]
        return [self.dataset[int(self.video_record_indices[int(column)])] for column in columns]
