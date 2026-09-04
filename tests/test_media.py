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
