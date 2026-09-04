from __future__ import annotations

import json
from pathlib import Path

from sceneseek.benchmarks import create_msrvtt_manifest


def test_create_msrvtt_manifest_preserves_explicit_split(tmp_path: Path) -> None:
    annotations = tmp_path / "annotations.json"
    annotations.write_text(
        json.dumps(
            {
                "sentences": [
                    {"video_id": "video0", "caption": "a person is walking"},
                    {"video_id": "video0", "caption": "someone walks"},
                    {"video_id": "video1", "caption": "a dog runs"},
                ]
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "manifest.jsonl"
    result = create_msrvtt_manifest(
        annotations,
        tmp_path / "videos",
        output,
        split="test",
    )
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert result["samples"] == 3
    assert result["videos"] == 2
    assert {row["split"] for row in rows} == {"test"}
    assert rows[0]["video_path"].endswith("video0.mp4")


def test_create_msrvtt_manifest_accepts_split_record_array(tmp_path: Path) -> None:
    annotations = tmp_path / "msrvtt_test_1k.json"
    annotations.write_text(
        json.dumps(
            [
                {
                    "video_id": "video7020",
                    "video": "video7020.mp4",
                    "caption": "a woman creates a fondant baby",
                },
                {
                    "video_id": "video7021",
                    "video": "video7021.mp4",
                    "caption": "a baseball player hits a ball",
                },
            ]
        ),
        encoding="utf-8",
    )
    output = tmp_path / "test.jsonl"
    result = create_msrvtt_manifest(
        annotations,
        tmp_path / "videos",
        output,
        split="test",
    )
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert result["samples"] == 2
    assert result["videos"] == 2
    assert [row["video_id"] for row in rows] == ["video7020", "video7021"]
    assert {row["split"] for row in rows} == {"test"}
    assert rows[0]["video_path"].endswith("video7020.mp4")


def test_create_msrvtt_manifest_expands_caption_lists(tmp_path: Path) -> None:
    annotations = tmp_path / "msrvtt_train_7k.json"
    annotations.write_text(
        json.dumps(
            [
                {
                    "video_id": "video0",
                    "video": "video0.mp4",
                    "caption": ["first caption", "second caption"],
                },
                {
                    "video_id": "video1",
                    "video": "video1.mp4",
                    "caption": ["third caption"],
                },
            ]
        ),
        encoding="utf-8",
    )
    output = tmp_path / "train.jsonl"
    result = create_msrvtt_manifest(
        annotations,
        tmp_path / "videos",
        output,
        split="train",
    )
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert result["samples"] == 3
    assert result["videos"] == 2
    assert [row["caption"] for row in rows] == [
        "first caption",
        "second caption",
        "third caption",
    ]
    assert rows[0]["sample_id"] != rows[1]["sample_id"]
    assert rows[0]["video_id"] == rows[1]["video_id"] == "video0"
