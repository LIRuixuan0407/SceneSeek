from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

FEATURE_PIPELINE_VERSION = "features-v2"
FRAME_PIPELINE_VERSION = "frames-v1"
PREPROCESSING_VERSION = "rgb-exif-v1"
VIDEO_SAMPLING_VERSION = "window-samples-v1"
TEMPORAL_AGGREGATION_VERSION = "mean-pool-v1"


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value is not None else default


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value is not None else default


def _fingerprint(payload: dict[str, object]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]


@dataclass(slots=True)
class Settings:
    data_dir: Path = field(
        default_factory=lambda: Path(os.getenv("SCENESEEK_DATA_DIR", "./data")).resolve()
    )
    encoder: str = field(default_factory=lambda: os.getenv("SCENESEEK_ENCODER", "auto"))
    model_id: str = field(
        default_factory=lambda: os.getenv("SCENESEEK_MODEL_ID", "openai/clip-vit-base-patch32")
    )
    device: str = field(default_factory=lambda: os.getenv("SCENESEEK_DEVICE", "auto"))
    encoder_batch_size: int = field(
        default_factory=lambda: _int_env("SCENESEEK_ENCODER_BATCH_SIZE", 32)
    )
    index_backend: str = field(default_factory=lambda: os.getenv("SCENESEEK_INDEX_BACKEND", "auto"))
    video_fps: float = field(default_factory=lambda: _float_env("SCENESEEK_VIDEO_FPS", 1.0))
    window_seconds: float = field(
        default_factory=lambda: _float_env("SCENESEEK_WINDOW_SECONDS", 4.0)
    )
    window_stride_seconds: float = field(
        default_factory=lambda: _float_env("SCENESEEK_WINDOW_STRIDE_SECONDS", 2.0)
    )
    max_frames_per_window: int = field(
        default_factory=lambda: _int_env("SCENESEEK_MAX_FRAMES_PER_WINDOW", 8)
    )
    temporal_checkpoint: Path | None = field(
        default_factory=lambda: (
            Path(value).expanduser().resolve()
            if (value := os.getenv("SCENESEEK_TEMPORAL_CHECKPOINT", "").strip())
            else None
        )
    )
    coarse_top_k: int = field(default_factory=lambda: _int_env("SCENESEEK_COARSE_TOP_K", 200))
    allowed_roots: tuple[Path, ...] = field(default_factory=tuple)
    cors_origins: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.allowed_roots:
            raw_roots = os.getenv("SCENESEEK_ALLOWED_ROOTS", "")
            self.allowed_roots = tuple(
                Path(item.strip()).expanduser().resolve()
                for item in raw_roots.split(",")
                if item.strip()
            )
        if not self.cors_origins:
            raw_origins = os.getenv("SCENESEEK_CORS_ORIGINS", "http://localhost:5173")
            self.cors_origins = tuple(
                item.strip() for item in raw_origins.split(",") if item.strip()
            )

        self.encoder_batch_size = max(1, self.encoder_batch_size)
        self.video_fps = max(0.01, self.video_fps)
        self.max_frames_per_window = max(1, self.max_frames_per_window)
        if self.temporal_checkpoint is not None and not self.temporal_checkpoint.is_file():
            raise ValueError(
                f"SCENESEEK_TEMPORAL_CHECKPOINT 不存在: {self.temporal_checkpoint}"
            )
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "cache" / "thumbnails").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "cache" / "clips").mkdir(parents=True, exist_ok=True)

    @property
    def database_path(self) -> Path:
        return self.data_dir / "sceneseek.sqlite3"

    @property
    def index_path(self) -> Path:
        return self.data_dir / "vectors.npz"

    @property
    def index_version(self) -> str:
        requested_encoder = f"{self.encoder}:{self.model_id}"
        return self.model_version_for(requested_encoder)

    @property
    def temporal_aggregation_version(self) -> str:
        if self.temporal_checkpoint is None:
            return TEMPORAL_AGGREGATION_VERSION
        digest = hashlib.sha256(self.temporal_checkpoint.read_bytes()).hexdigest()[:12]
        return f"temporal-adapter:{digest}"

    def model_version_for(self, encoder_version: str) -> str:
        payload: dict[str, object] = {
            "pipeline": FEATURE_PIPELINE_VERSION,
            "encoder": encoder_version,
            "preprocessing": PREPROCESSING_VERSION,
            "video_sampling": VIDEO_SAMPLING_VERSION,
            "video_fps": self.video_fps,
            "window_seconds": self.window_seconds,
            "window_stride_seconds": self.window_stride_seconds,
            "max_frames_per_window": self.max_frames_per_window,
            "temporal_aggregation": self.temporal_aggregation_version,
        }
        return f"{FEATURE_PIPELINE_VERSION}:{encoder_version}:{_fingerprint(payload)}"

    def frame_embedding_version_for(self, encoder_version: str) -> str:
        payload: dict[str, object] = {
            "pipeline": FRAME_PIPELINE_VERSION,
            "encoder": encoder_version,
            "preprocessing": PREPROCESSING_VERSION,
        }
        return f"{FRAME_PIPELINE_VERSION}:{encoder_version}:{_fingerprint(payload)}"

    def assert_allowed_root(self, path: Path) -> Path:
        resolved = path.expanduser().resolve()
        if not resolved.exists() or not resolved.is_dir():
            raise ValueError(f"媒体目录不存在或不是目录: {resolved}")
        if self.allowed_roots and not any(
            resolved == root or resolved.is_relative_to(root) for root in self.allowed_roots
        ):
            raise PermissionError("该目录不在 SCENESEEK_ALLOWED_ROOTS 允许范围内")
        return resolved
