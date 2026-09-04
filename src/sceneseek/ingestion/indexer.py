from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from sceneseek.config import Settings
from sceneseek.encoders import Encoder
from sceneseek.encoders.base import l2_normalize
from sceneseek.ingestion.media import create_thumbnail, extract_video_frame, load_image
from sceneseek.storage import Database

ProgressCallback = Callable[[int, int, str], None]


class MediaIndexer:
    def __init__(self, settings: Settings, database: Database, encoder: Encoder) -> None:
        self.settings = settings
        self.database = database
        self.encoder = encoder
        self.model_version = (
            f"{encoder.version}:w{settings.window_seconds:g}:s{settings.window_stride_seconds:g}"
        )

    def build(
        self, *, rebuild: bool = False, progress: ProgressCallback | None = None
    ) -> dict[str, object]:
        media = self.database.list_media(needs_version=None if rebuild else self.model_version)
        indexed = 0
        error_items: list[dict[str, str]] = []
        for position, record in enumerate(media, start=1):
            if progress:
                progress(position - 1, len(media), Path(record.path).name)
            try:
                self.database.invalidate_media(record.media_id)
                if record.media_type == "image":
                    self._index_image(record.media_id, Path(record.path))
                else:
                    self._index_video(record.media_id, Path(record.path))
                self.database.mark_indexed(record.media_id, self.model_version)
                indexed += 1
            except Exception as error:  # noqa: BLE001 - record one media failure and continue
                error_items.append(
                    {"media_id": record.media_id, "path": record.path, "error": str(error)}
                )
        if progress:
            progress(len(media), len(media), "索引完成")
        return {
            "processed": len(media),
            "indexed": indexed,
            "errors": len(error_items),
            "error_items": error_items,
            "model_version": self.model_version,
        }

    def _index_image(self, media_id: str, path: Path) -> None:
        image = load_image(path)
        vector = self.encoder.encode_image(image, context=path.stem)
        self.database.put_embedding(
            item_id=media_id,
            media_id=media_id,
            kind="image",
            vector=vector,
            model_version=self.model_version,
        )
        thumbnail = self.settings.data_dir / "cache" / "thumbnails" / f"{media_id}.jpg"
        create_thumbnail(image, thumbnail)

    def _index_video(self, media_id: str, path: Path) -> None:
        clips = self.database.list_clips(media_id)
        if not clips:
            raise ValueError("视频没有可索引窗口")
        for clip in clips:
            frames = [extract_video_frame(path, timestamp) for timestamp in clip.sampled_frame_ts]
            vectors = [self.encoder.encode_image(frame, context=path.stem) for frame in frames]
            vector = l2_normalize(np.mean(np.stack(vectors), axis=0))
            self.database.put_embedding(
                item_id=clip.clip_id,
                media_id=media_id,
                kind="clip",
                vector=vector,
                model_version=self.model_version,
                start_sec=clip.start_sec,
                end_sec=clip.end_sec,
            )

        middle = clips[len(clips) // 2]
        frame = extract_video_frame(path, middle.thumbnail_sec)
        thumbnail = self.settings.data_dir / "cache" / "thumbnails" / f"{media_id}.jpg"
        create_thumbnail(frame, thumbnail)
