from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value is not None else default


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value is not None else default


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
        safe_model = self.model_id.replace("/", "--")
        return (
            f"{self.encoder}:{safe_model}:w{self.window_seconds:g}:s{self.window_stride_seconds:g}"
        )

    def assert_allowed_root(self, path: Path) -> Path:
        resolved = path.expanduser().resolve()
        if not resolved.exists() or not resolved.is_dir():
            raise ValueError(f"媒体目录不存在或不是目录: {resolved}")
        if self.allowed_roots and not any(
            resolved == root or resolved.is_relative_to(root) for root in self.allowed_roots
        ):
            raise PermissionError("该目录不在 SCENESEEK_ALLOWED_ROOTS 允许范围内")
        return resolved
