from __future__ import annotations

import hashlib
import io
import json
import math
import re
import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageOps

from sceneseek.domain import ClipRecord

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v", ".mpeg", ".mpg"}


def media_type_for(path: Path) -> str | None:
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    if suffix in VIDEO_EXTENSIONS:
        return "video"
    return None


def content_hash(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_media_id(path: Path) -> str:
    normalized = str(path.resolve()).encode("utf-8")
    return "med_" + hashlib.sha256(normalized).hexdigest()[:20]


def probe_image(path: Path) -> dict[str, int | float | None]:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        width, height = image.size
    return {"width": width, "height": height, "duration": None, "fps": None}


def probe_video(path: Path) -> dict[str, int | float | None]:
    if shutil.which("ffprobe") is None:
        return _probe_video_with_ffmpeg(path)
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate:format=duration",
        "-of",
        "json",
        str(path),
    ]
    process = subprocess.run(command, capture_output=True, check=True, timeout=30)
    payload = json.loads(process.stdout)
    if not payload.get("streams"):
        raise ValueError("视频不包含可解码的视频流")
    stream = payload["streams"][0]
    numerator, denominator = (stream.get("avg_frame_rate") or "0/1").split("/")
    fps = float(numerator) / max(float(denominator), 1.0)
    return {
        "width": int(stream.get("width") or 0) or None,
        "height": int(stream.get("height") or 0) or None,
        "duration": float(payload.get("format", {}).get("duration") or 0.0),
        "fps": fps or None,
    }


def _probe_video_with_ffmpeg(path: Path) -> dict[str, int | float | None]:
    process = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", str(path)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    output = process.stderr
    duration_match = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", output)
    video_line = next((line for line in output.splitlines() if "Video:" in line), "")
    size_match = re.search(r"(?:^|[ ,])(\d{2,5})x(\d{2,5})(?:[ ,\[])", video_line)
    fps_match = re.search(r"([\d.]+)\s+fps", video_line)
    if not video_line or not size_match:
        raise ValueError("视频不包含可解码的视频流")
    duration = None
    if duration_match:
        hours, minutes, seconds = duration_match.groups()
        duration = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    return {
        "width": int(size_match.group(1)),
        "height": int(size_match.group(2)),
        "duration": duration,
        "fps": float(fps_match.group(1)) if fps_match else None,
    }


def build_clips(
    media_id: str,
    duration: float,
    *,
    window_seconds: float,
    stride_seconds: float,
    sample_fps: float,
    max_frames: int,
) -> list[ClipRecord]:
    if duration <= 0:
        return []
    window_seconds = max(window_seconds, 0.25)
    stride_seconds = max(stride_seconds, 0.25)
    starts: list[float] = []
    start = 0.0
    while start < duration:
        starts.append(start)
        if start + window_seconds >= duration:
            break
        start += stride_seconds

    records: list[ClipRecord] = []
    for index, start in enumerate(starts):
        end = min(duration, start + window_seconds)
        frame_count = max(1, min(max_frames, math.ceil((end - start) * sample_fps)))
        step = (end - start) / frame_count
        timestamps = tuple(
            round(min(end - 0.001, start + step * (offset + 0.5)), 3)
            for offset in range(frame_count)
        )
        records.append(
            ClipRecord(
                clip_id=f"{media_id}_clip_{index:06d}",
                media_id=media_id,
                start_sec=round(start, 3),
                end_sec=round(end, 3),
                thumbnail_sec=round((start + end) / 2, 3),
                sampled_frame_ts=timestamps,
            )
        )
    return records


def load_image(path: Path) -> Image.Image:
    with Image.open(path) as source:
        return ImageOps.exif_transpose(source).convert("RGB")


def extract_video_frame(path: Path, timestamp: float) -> Image.Image:
    command = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-ss",
        f"{max(timestamp, 0):.3f}",
        "-i",
        str(path),
        "-frames:v",
        "1",
        "-f",
        "image2pipe",
        "-vcodec",
        "png",
        "pipe:1",
    ]
    process = subprocess.run(command, capture_output=True, check=True, timeout=45)
    if not process.stdout:
        raise ValueError(f"无法在 {timestamp:.2f}s 解码视频帧")
    with Image.open(io.BytesIO(process.stdout)) as image:
        return image.convert("RGB")


def create_thumbnail(image: Image.Image, output: Path, size: tuple[int, int] = (720, 480)) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    thumbnail = image.copy()
    thumbnail.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, (18, 21, 26))
    x = (size[0] - thumbnail.width) // 2
    y = (size[1] - thumbnail.height) // 2
    canvas.paste(thumbnail, (x, y))
    canvas.save(output, format="JPEG", quality=86, optimize=True)
    return output
