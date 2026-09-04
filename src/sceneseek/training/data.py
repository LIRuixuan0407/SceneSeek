from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True, frozen=True)
class TemporalSample:
    sample_id: str
    video_id: str
    caption: str
    split: str
    video_feature_file: str
    text_index: int


def _safe_video_filename(video_id: str) -> str:
    digest = hashlib.sha256(video_id.encode("utf-8")).hexdigest()[:16]
    return f"{digest}.npz"


def _sample_video_timestamps(
    duration: float, sample_fps: float, max_frames: int
) -> list[float]:
    frame_count = max(1, min(max_frames, int(np.ceil(duration * sample_fps))))
    step = duration / frame_count
    return [
        round(min(max(duration - 0.001, 0.0), step * (index + 0.5)), 3)
        for index in range(frame_count)
    ]


def read_source_manifest(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            for field in ("sample_id", "video_id", "video_path", "caption"):
                if not payload.get(field):
                    raise ValueError(f"{path}:{line_number} 缺少字段 {field}")
            payload.setdefault("split", "train")
            records.append(payload)
    if not records:
        raise ValueError("manifest 为空")
    return records


def prepare_temporal_features(
    source_manifest: Path,
    output_dir: Path,
    encoder,
    *,
    sample_fps: float = 1.0,
    max_frames: int = 16,
    batch_size: int = 32,
    overwrite: bool = False,
) -> dict[str, int | str]:
    from sceneseek.ingestion.media import extract_video_frames, probe_video

    if sample_fps <= 0:
        raise ValueError("sample_fps 必须大于 0")
    if max_frames <= 0:
        raise ValueError("max_frames 必须大于 0")

    records = read_source_manifest(source_manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_dir = output_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    grouped: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for record in records:
        video_id = str(record["video_id"])
        path = Path(record["video_path"]).expanduser().resolve()
        previous = grouped.get(video_id)
        if previous is not None and Path(previous["video_path"]).resolve() != path:
            raise ValueError(f"video_id={video_id} 对应多个不同文件")
        grouped.setdefault(video_id, {**record, "video_path": str(path)})

    video_files: dict[str, str] = {}
    videos_created = 0
    for video_id, record in grouped.items():
        source = Path(record["video_path"])
        if not source.is_file():
            raise FileNotFoundError(source)
        relative = Path("videos") / _safe_video_filename(video_id)
        target = output_dir / relative
        video_files[video_id] = relative.as_posix()
        if target.exists() and not overwrite:
            continue

        metadata = probe_video(source)
        duration = float(metadata.get("duration") or 0.0)
        if duration <= 0:
            raise ValueError(f"视频时长无效: {source}")
        timestamps = _sample_video_timestamps(duration, sample_fps, max_frames)
        decoded = extract_video_frames(source, timestamps)
        images = [decoded[round(value, 3)] for value in timestamps]
        vectors = encoder.encode_images(images, batch_size=batch_size)
        np.savez_compressed(
            target,
            frames=np.asarray(vectors, dtype=np.float32),
            timestamps=np.asarray(timestamps, dtype=np.float32),
        )
        videos_created += 1

    captions = [str(record["caption"]) for record in records]
    text_vectors = encoder.encode_texts(captions, batch_size=batch_size)
    np.save(output_dir / "texts.npy", np.asarray(text_vectors, dtype=np.float32))

    feature_manifest = output_dir / "manifest.jsonl"
    with feature_manifest.open("w", encoding="utf-8") as stream:
        for text_index, record in enumerate(records):
            sample = {
                "sample_id": str(record["sample_id"]),
                "video_id": str(record["video_id"]),
                "caption": str(record["caption"]),
                "split": str(record.get("split") or "train"),
                "video_feature_file": video_files[str(record["video_id"])],
                "text_index": text_index,
            }
            stream.write(json.dumps(sample, ensure_ascii=False) + "\n")

    metadata = {
        "encoder_version": encoder.version,
        "dimension": int(encoder.dimension),
        "sample_fps": float(sample_fps),
        "max_frames": int(max_frames),
        "samples": len(records),
        "videos": len(grouped),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "samples": len(records),
        "videos": len(grouped),
        "videos_created": videos_created,
        "feature_dir": str(output_dir.resolve()),
    }


class TemporalFeatureDataset:
    def __init__(self, root: Path, *, split: str | None = None, cache_size: int = 128) -> None:
        self.root = root.resolve()
        self.texts = np.load(self.root / "texts.npy", mmap_mode="r")
        self.records: list[TemporalSample] = []
        with (self.root / "manifest.jsonl").open("r", encoding="utf-8") as stream:
            for raw in stream:
                if not raw.strip():
                    continue
                payload = json.loads(raw)
                if split is not None and payload["split"] != split:
                    continue
                self.records.append(TemporalSample(**payload))
        if not self.records:
            raise ValueError(f"feature store 中没有 split={split!r} 的样本")
        self.cache_size = max(1, cache_size)
        self._video_cache: OrderedDict[str, np.ndarray] = OrderedDict()

    def __len__(self) -> int:
        return len(self.records)

    def _load_video(self, relative: str) -> np.ndarray:
        cached = self._video_cache.get(relative)
        if cached is not None:
            self._video_cache.move_to_end(relative)
            return cached
        with np.load(self.root / relative) as payload:
            frames = np.asarray(payload["frames"], dtype=np.float32)
        self._video_cache[relative] = frames
        if len(self._video_cache) > self.cache_size:
            self._video_cache.popitem(last=False)
        return frames

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        return {
            "sample_id": record.sample_id,
            "video_id": record.video_id,
            "caption": record.caption,
            "frames": self._load_video(record.video_feature_file),
            "text": np.asarray(self.texts[record.text_index], dtype=np.float32),
        }

    @property
    def dimension(self) -> int:
        return int(self.texts.shape[1])


def collate_temporal_batch(items: list[dict[str, object]]) -> dict[str, object]:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("请先安装 `pip install -e '.[ml]'`") from error

    max_steps = max(np.asarray(item["frames"]).shape[0] for item in items)
    dimension = np.asarray(items[0]["frames"]).shape[1]
    frames = torch.zeros((len(items), max_steps, dimension), dtype=torch.float32)
    mask = torch.zeros((len(items), max_steps), dtype=torch.bool)
    texts = torch.zeros((len(items), dimension), dtype=torch.float32)
    video_ids: list[str] = []
    for row, item in enumerate(items):
        array = np.asarray(item["frames"], dtype=np.float32)
        steps = array.shape[0]
        frames[row, :steps] = torch.from_numpy(array)
        mask[row, :steps] = True
        text = np.array(item["text"], dtype=np.float32, copy=True)
        texts[row] = torch.from_numpy(text)
        video_ids.append(str(item["video_id"]))
    return {"frames": frames, "mask": mask, "texts": texts, "video_ids": video_ids}
