from __future__ import annotations

import time
from pathlib import Path

from PIL import Image

from sceneseek.config import Settings
from sceneseek.ingestion import MediaScanner
from sceneseek.storage import Database


def settings_for(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        encoder="lite",
        allowed_roots=(tmp_path,),
        cors_origins=("http://test",),
    )


def test_scan_is_incremental_and_invalidates_changed_media(tmp_path: Path) -> None:
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    red = media_dir / "red.jpg"
    blue = media_dir / "blue.png"
    Image.new("RGB", (30, 20), "red").save(red)
    Image.new("RGB", (12, 18), "blue").save(blue)
    settings = settings_for(tmp_path)
    database = Database(settings.database_path)
    scanner = MediaScanner(settings, database)

    first = scanner.scan(media_dir)
    second = scanner.scan(media_dir)
    time.sleep(0.02)
    Image.new("RGB", (40, 20), "green").save(red)
    third = scanner.scan(media_dir)

    assert (first.added, first.errors) == (2, 0)
    assert second.unchanged == 2
    assert third.updated == 1
    assert database.get_media_by_path(str(red.resolve())).width == 40  # type: ignore[union-attr]


def test_scan_removes_deleted_media(tmp_path: Path) -> None:
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    image_path = media_dir / "one.jpg"
    Image.new("RGB", (10, 10), "white").save(image_path)
    settings = settings_for(tmp_path)
    database = Database(settings.database_path)
    scanner = MediaScanner(settings, database)
    scanner.scan(media_dir)
    record = database.get_media_by_path(str(image_path.resolve()))
    assert record is not None
    thumbnail = settings.data_dir / "cache" / "thumbnails" / f"{record.media_id}.jpg"
    thumbnail.write_bytes(b"cached")

    image_path.unlink()
    result = scanner.scan(media_dir)

    assert result.removed == 1
    assert database.counts()["images"] == 0
    assert not thumbnail.exists()


def test_changed_file_becoming_duplicate_removes_stale_media(tmp_path: Path) -> None:
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    first_path = media_dir / "first.png"
    second_path = media_dir / "second.png"
    Image.new("RGB", (20, 20), "red").save(first_path)
    Image.new("RGB", (20, 20), "blue").save(second_path)
    settings = settings_for(tmp_path)
    database = Database(settings.database_path)
    scanner = MediaScanner(settings, database)

    scanner.scan(media_dir)
    stale = database.get_media_by_path(str(second_path.resolve()))
    assert stale is not None

    time.sleep(0.02)
    second_path.write_bytes(first_path.read_bytes())
    result = scanner.scan(media_dir)

    assert result.duplicates == 1
    assert database.get_media_by_path(str(second_path.resolve())) is None
    assert database.counts()["images"] == 1


def test_unchanged_video_refreshes_clips_when_sampling_config_changes(tmp_path: Path) -> None:
    import shutil
    import subprocess

    if shutil.which("ffmpeg") is None:
        return
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    video_path = media_dir / "sample.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=160x120:d=5",
            "-pix_fmt",
            "yuv420p",
            "-y",
            str(video_path),
        ],
        check=True,
        timeout=30,
    )
    first_settings = Settings(
        data_dir=tmp_path / "data",
        encoder="lite",
        window_seconds=4.0,
        window_stride_seconds=2.0,
        allowed_roots=(tmp_path,),
        cors_origins=("http://test",),
    )
    database = Database(first_settings.database_path)
    first_scanner = MediaScanner(first_settings, database)
    first_scanner.scan(media_dir)
    media = database.get_media_by_path(str(video_path.resolve()))
    assert media is not None
    database.mark_indexed(media.media_id, "old-version")
    assert len(database.list_clips(media.media_id)) == 2

    second_settings = Settings(
        data_dir=tmp_path / "data",
        encoder="lite",
        window_seconds=2.0,
        window_stride_seconds=1.0,
        allowed_roots=(tmp_path,),
        cors_origins=("http://test",),
    )
    result = MediaScanner(second_settings, database).scan(media_dir)

    refreshed = database.get_media(media.media_id)
    assert result.updated == 1
    assert refreshed is not None and refreshed.index_version is None
    assert len(database.list_clips(media.media_id)) == 4
