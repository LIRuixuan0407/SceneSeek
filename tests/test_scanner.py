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
