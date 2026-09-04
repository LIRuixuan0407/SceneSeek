from __future__ import annotations

from sceneseek.ingestion.media import build_clips


def test_clip_windows_cover_video_tail() -> None:
    clips = build_clips(
        "video-1",
        9.0,
        window_seconds=4.0,
        stride_seconds=2.0,
        sample_fps=1.0,
        max_frames=8,
    )

    assert [(clip.start_sec, clip.end_sec) for clip in clips] == [
        (0.0, 4.0),
        (2.0, 6.0),
        (4.0, 8.0),
        (6.0, 9.0),
    ]
    assert clips[-1].sampled_frame_ts[-1] < 9.0


def test_extract_video_frames_retries_slightly_earlier_when_bulk_output_is_missing(
    monkeypatch,
) -> None:
    from pathlib import Path
    from types import SimpleNamespace

    from PIL import Image

    from sceneseek.ingestion import media

    monkeypatch.setattr(
        media.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=b"", stderr=b""),
    )

    attempts: list[float] = []

    def fake_extract_video_frame(path: Path, timestamp: float) -> Image.Image:
        attempts.append(timestamp)
        if timestamp == 1.0:
            raise ValueError("tail frame unavailable")
        return Image.new("RGB", (2, 2))

    monkeypatch.setattr(media, "extract_video_frame", fake_extract_video_frame)

    frames = media.extract_video_frames(Path("video.mp4"), [1.0], batch_size=1)

    assert attempts == [1.0, 0.95]
    assert frames[1.0].size == (2, 2)
