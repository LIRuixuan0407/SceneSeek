from __future__ import annotations

import json
from pathlib import Path


def create_msrvtt_manifest(
    annotations: Path,
    video_root: Path,
    output: Path,
    *,
    split: str,
    extension: str = ".mp4",
) -> dict[str, int | str]:
    """Convert the common MSR-VTT JSON format into SceneSeek's generic JSONL manifest.

    The caller supplies the split explicitly because public MSR-VTT distributions use
    multiple split conventions/files. SceneSeek deliberately does not silently invent one.
    """
    payload = json.loads(annotations.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        sentences = payload.get("sentences")
        if not isinstance(sentences, list):
            raise ValueError("MSR-VTT annotations 缺少 sentences 数组")
    elif isinstance(payload, list):
        sentences = payload
    else:
        raise ValueError("MSR-VTT annotations 必须是 sentences 对象或记录数组")

    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    unique: set[str] = set()
    with output.open("w", encoding="utf-8") as stream:
        for index, sentence in enumerate(sentences):
            if not isinstance(sentence, dict):
                continue
            video_id = str(sentence.get("video_id") or "").strip()
            raw_caption = sentence.get("caption")
            if isinstance(raw_caption, list):
                captions = [str(value).strip() for value in raw_caption if str(value).strip()]
            else:
                caption = str(raw_caption or "").strip()
                captions = [caption] if caption else []
            if not video_id or not captions:
                continue
            video_path = (video_root / f"{video_id}{extension}").resolve()
            for caption_index, caption in enumerate(captions):
                suffix = f":{caption_index}" if len(captions) > 1 else ""
                record = {
                    "sample_id": f"{split}:{video_id}:{index}{suffix}",
                    "video_id": video_id,
                    "video_path": str(video_path),
                    "caption": caption,
                    "split": split,
                }
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
                unique.add(video_id)
    if count == 0:
        raise ValueError("annotations 中没有可用 caption")
    return {"samples": count, "videos": len(unique), "output": str(output.resolve())}
