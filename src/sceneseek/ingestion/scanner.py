from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from sceneseek.config import Settings
from sceneseek.domain import ClipRecord, MediaRecord, ScanSummary
from sceneseek.ingestion.media import (
    build_clips,
    content_hash,
    media_type_for,
    probe_image,
    probe_video,
    stable_media_id,
)
from sceneseek.storage import Database

ProgressCallback = Callable[[int, int, str], None]


class MediaScanner:
    def __init__(self, settings: Settings, database: Database) -> None:
        self.settings = settings
        self.database = database

    def scan(self, root: Path, progress: ProgressCallback | None = None) -> ScanSummary:
        root = self.settings.assert_allowed_root(root)
        files = sorted(
            path for path in root.rglob("*") if path.is_file() and media_type_for(path) is not None
        )
        present_paths = {str(path.resolve()) for path in files}
        summary = ScanSummary(root=str(root), discovered=len(files))
        removed_ids = self.database.delete_missing_under(root, present_paths)
        summary.removed = len(removed_ids)
        self._remove_cached_media(removed_ids)

        for position, path in enumerate(files, start=1):
            if progress:
                progress(position - 1, len(files), path.name)
            try:
                self._scan_file(path, summary)
            except Exception:  # noqa: BLE001 - one broken file must not abort a library scan
                summary.errors += 1

        if progress:
            progress(len(files), len(files), "扫描完成")
        return summary

    def _scan_file(self, path: Path, summary: ScanSummary) -> None:
        resolved = path.resolve()
        path_text = str(resolved)
        stat = resolved.stat()
        existing = self.database.get_media_by_path(path_text)
        if existing and existing.mtime == stat.st_mtime and existing.size_bytes == stat.st_size:
            if existing.media_type == "video" and self._refresh_video_clips_if_needed(existing):
                summary.updated += 1
            else:
                summary.unchanged += 1
            return

        digest = content_hash(resolved)
        duplicate = self.database.find_by_hash(digest, exclude_path=path_text)
        if duplicate is not None:
            if existing is not None:
                self.database.delete_media(existing.media_id)
                self._remove_cached_media([existing.media_id])
            summary.duplicates += 1
            return

        media_type = media_type_for(resolved)
        if media_type is None:
            return
        metadata = probe_image(resolved) if media_type == "image" else probe_video(resolved)
        media_id = existing.media_id if existing else stable_media_id(resolved)
        record = MediaRecord(
            media_id=media_id,
            path=path_text,
            media_type=media_type,  # type: ignore[arg-type]
            duration=metadata["duration"],
            fps=metadata["fps"],
            width=metadata["width"],  # type: ignore[arg-type]
            height=metadata["height"],  # type: ignore[arg-type]
            mtime=stat.st_mtime,
            size_bytes=stat.st_size,
            content_hash=digest,
            index_version=None,
        )
        if existing:
            self.database.invalidate_media(media_id, drop_frame_cache=True)
            summary.updated += 1
        else:
            summary.added += 1
        self.database.upsert_media(record)

        if media_type == "video":
            self.database.replace_clips(media_id, self._build_video_clips(record))

    def _refresh_video_clips_if_needed(self, record: MediaRecord) -> bool:
        expected = self._build_video_clips(record)
        current = self.database.list_clips(record.media_id)
        if current == expected:
            return False
        self.database.replace_clips(record.media_id, expected)
        self.database.invalidate_media(record.media_id)
        return True

    def _build_video_clips(self, record: MediaRecord) -> list[ClipRecord]:
        return build_clips(
            record.media_id,
            float(record.duration or 0),
            window_seconds=self.settings.window_seconds,
            stride_seconds=self.settings.window_stride_seconds,
            sample_fps=self.settings.video_fps,
            max_frames=self.settings.max_frames_per_window,
        )

    def _remove_cached_media(self, media_ids: list[str]) -> None:
        thumbnail_dir = self.settings.data_dir / "cache" / "thumbnails"
        clip_dir = self.settings.data_dir / "cache" / "clips"
        for media_id in media_ids:
            for cached in thumbnail_dir.glob(f"{media_id}*.jpg"):
                cached.unlink(missing_ok=True)
            for cached in clip_dir.glob(f"{media_id}-*.mp4"):
                cached.unlink(missing_ok=True)
