from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from sceneseek.config import Settings
from sceneseek.encoders.lite import LiteEncoder
from sceneseek.ingestion import MediaIndexer, MediaScanner
from sceneseek.retrieval import RetrievalService, VectorIndex
from sceneseek.storage import Database


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required")
def test_video_is_split_indexed_and_returned_as_a_moment(tmp_path: Path) -> None:
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    video_path = media_dir / "blue-ocean.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=320x240:d=5",
            "-pix_fmt",
            "yuv420p",
            "-y",
            str(video_path),
        ],
        check=True,
        timeout=30,
    )
    settings = Settings(
        data_dir=tmp_path / "state",
        encoder="lite",
        index_backend="numpy",
        allowed_roots=(tmp_path,),
        cors_origins=("http://test",),
    )
    database = Database(settings.database_path)
    scanner = MediaScanner(settings, database)
    encoder = LiteEncoder()
    indexer = MediaIndexer(settings, database, encoder)
    index = VectorIndex(settings.index_path, "numpy")

    scan = scanner.scan(media_dir)
    build = indexer.build()
    index.rebuild(database, indexer.model_version)
    response = RetrievalService(settings, encoder, index).search_text("blue ocean")

    assert scan.added == 1
    assert database.counts()["clips"] == 2
    assert build["errors"] == 0
    assert response["results"][0]["media_type"] == "video"  # type: ignore[index]
    assert response["results"][0]["start_sec"] == 0.0  # type: ignore[index]
    assert response["results"][0]["end_sec"] == pytest.approx(5.0)  # type: ignore[index]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required")
def test_video_frame_embeddings_are_reused_on_rebuild(tmp_path: Path) -> None:
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    video_path = media_dir / "cached.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=160x120:d=5",
            "-pix_fmt",
            "yuv420p",
            "-y",
            str(video_path),
        ],
        check=True,
        timeout=30,
    )
    settings = Settings(
        data_dir=tmp_path / "state",
        encoder="lite",
        index_backend="numpy",
        allowed_roots=(tmp_path,),
        cors_origins=("http://test",),
    )
    database = Database(settings.database_path)
    MediaScanner(settings, database).scan(media_dir)
    indexer = MediaIndexer(settings, database, LiteEncoder())

    first = indexer.build(rebuild=True)
    second = indexer.build(rebuild=True)

    assert first["frame_cache_misses"] > 0  # type: ignore[operator]
    assert first["frame_cache_hits"] == 0
    assert second["frame_cache_misses"] == 0
    assert second["frame_cache_hits"] == first["frame_cache_misses"]
