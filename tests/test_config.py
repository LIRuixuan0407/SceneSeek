from __future__ import annotations

from pathlib import Path

from sceneseek.config import Settings
from sceneseek.encoders.lite import LiteEncoder
from sceneseek.ingestion import MediaIndexer
from sceneseek.storage import Database


def test_model_version_captures_video_feature_configuration(tmp_path: Path) -> None:
    common = {
        "data_dir": tmp_path / "state",
        "encoder": "lite",
        "allowed_roots": (tmp_path,),
        "cors_origins": ("http://test",),
    }
    first = Settings(**common, video_fps=1.0, max_frames_per_window=8)
    second = Settings(**common, video_fps=2.0, max_frames_per_window=16)

    first_version = MediaIndexer(first, Database(first.database_path), LiteEncoder()).model_version
    second_version = MediaIndexer(
        second, Database(second.database_path), LiteEncoder()
    ).model_version

    assert first_version != second_version


def test_frame_cache_version_is_independent_of_window_layout(tmp_path: Path) -> None:
    common = {
        "data_dir": tmp_path / "state",
        "encoder": "lite",
        "allowed_roots": (tmp_path,),
        "cors_origins": ("http://test",),
    }
    first = Settings(**common, window_seconds=4.0, window_stride_seconds=2.0)
    second = Settings(**common, window_seconds=8.0, window_stride_seconds=4.0)
    encoder = LiteEncoder()

    assert first.frame_embedding_version_for(encoder.version) == second.frame_embedding_version_for(
        encoder.version
    )
    assert first.model_version_for(encoder.version) != second.model_version_for(encoder.version)


def test_temporal_checkpoint_changes_index_version(tmp_path: Path) -> None:
    checkpoint = tmp_path / "adapter.pt"
    checkpoint.write_bytes(b"first")
    first = Settings(data_dir=tmp_path / "a", temporal_checkpoint=checkpoint)
    first_version = first.model_version_for("encoder:v1")

    checkpoint.write_bytes(b"second")
    second = Settings(data_dir=tmp_path / "b", temporal_checkpoint=checkpoint)
    second_version = second.model_version_for("encoder:v1")

    assert first_version != second_version
    assert first.frame_embedding_version_for("encoder:v1") == second.frame_embedding_version_for(
        "encoder:v1"
    )
