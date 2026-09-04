from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

MediaType = Literal["image", "video"]


@dataclass(slots=True)
class MediaRecord:
    media_id: str
    path: str
    media_type: MediaType
    duration: float | None
    fps: float | None
    width: int | None
    height: int | None
    mtime: float
    size_bytes: int
    content_hash: str
    index_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ClipRecord:
    clip_id: str
    media_id: str
    start_sec: float
    end_sec: float
    thumbnail_sec: float
    sampled_frame_ts: tuple[float, ...]


@dataclass(slots=True)
class SearchResult:
    media_id: str
    media_type: MediaType
    score: float
    path: str
    width: int | None = None
    height: int | None = None
    duration: float | None = None
    start_sec: float | None = None
    end_sec: float | None = None
    thumbnail_sec: float | None = None
    coarse_score: float | None = None
    rerank_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ScanSummary:
    root: str
    discovered: int = 0
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    duplicates: int = 0
    removed: int = 0
    errors: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
