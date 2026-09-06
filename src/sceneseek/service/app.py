from __future__ import annotations

import io
import mimetypes
import os
import subprocess
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from sceneseek.config import Settings
from sceneseek.encoders import create_encoder
from sceneseek.ingestion import MediaIndexer, MediaScanner
from sceneseek.ingestion.media import create_thumbnail, extract_video_frame, load_image
from sceneseek.retrieval import RetrievalService, VectorIndex
from sceneseek.service.jobs import JobManager
from sceneseek.service.schemas import BuildRequest, FeedbackRequest, ScanRequest, TextSearchRequest
from sceneseek.storage import Database


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    database = Database(settings.database_path)
    encoder = create_encoder(settings)
    indexer = MediaIndexer(settings, database, encoder)
    vector_index = VectorIndex(settings.index_path, settings.index_backend)
    retrieval = RetrievalService(settings, encoder, vector_index, database)
    scanner = MediaScanner(settings, database)
    jobs = JobManager()

    app = FastAPI(
        title="SceneSeek API",
        version="0.1.0",
        description="Local-first multimodal image and video retrieval with temporal moments.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.settings = settings
    app.state.database = database
    app.state.encoder = encoder
    app.state.indexer = indexer
    app.state.vector_index = vector_index
    app.state.retrieval = retrieval
    app.state.scanner = scanner
    app.state.jobs = jobs

    @app.get("/api/health")
    def health() -> dict[str, object]:
        return {"status": "ok", "encoder": encoder.name, "index_size": vector_index.size}

    @app.get("/api/index/status")
    def index_status() -> dict[str, object]:
        return {
            **jobs.snapshot(),
            "library": database.counts(),
            "index": {
                "size": vector_index.size,
                "dimension": vector_index.dimension,
                "backend": vector_index.backend,
                "model_version": indexer.model_version,
            },
            "encoder": {"name": encoder.name, "version": encoder.version},
        }

    @app.post("/api/index/scan", status_code=202)
    def scan(request: ScanRequest) -> dict[str, str]:
        def callback() -> dict[str, object]:
            summary = scanner.scan(Path(request.path), jobs.update_progress)
            vector_index.rebuild(database, indexer.model_version)
            return summary.to_dict()

        try:
            settings.assert_allowed_root(Path(request.path))
            jobs.start("scan", callback)
        except (ValueError, PermissionError, RuntimeError) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        return {"status": "accepted", "operation": "scan"}

    @app.post("/api/index/build", status_code=202)
    def build(request: BuildRequest) -> dict[str, str]:
        def callback() -> dict[str, object]:
            result = indexer.build(rebuild=request.rebuild, progress=jobs.update_progress)
            result["vectors"] = vector_index.rebuild(database, indexer.model_version)
            return result

        try:
            jobs.start("build", callback)
        except RuntimeError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        return {"status": "accepted", "operation": "build"}

    @app.post("/api/search/text")
    def search_text(request: TextSearchRequest) -> dict[str, object]:
        try:
            response = retrieval.search_text(
                request.query, limit=request.limit, media_type=request.media_type
            )
            return _with_urls(response)
        except (ValueError, RuntimeError) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.post("/api/search/image")
    async def search_image(
        file: Annotated[UploadFile, File()],
        limit: Annotated[int, Form(ge=1, le=100)] = 24,
        media_type: Annotated[str | None, Form()] = None,
    ) -> dict[str, object]:
        image = await _read_upload(file)
        try:
            return _with_urls(retrieval.search_image(image, limit=limit, media_type=media_type))
        except (ValueError, RuntimeError) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.post("/api/search/composed")
    async def search_composed(
        file: Annotated[UploadFile, File()],
        text: Annotated[str, Form(min_length=1)],
        text_weight: Annotated[float, Form(ge=0, le=1)] = 0.55,
        limit: Annotated[int, Form(ge=1, le=100)] = 24,
        media_type: Annotated[str | None, Form()] = None,
    ) -> dict[str, object]:
        image = await _read_upload(file)
        try:
            response = retrieval.search_composed(
                text,
                image,
                text_weight=text_weight,
                limit=limit,
                media_type=media_type,
            )
            return _with_urls(response)
        except (ValueError, RuntimeError) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.get("/api/media/{media_id}")
    def get_media(media_id: str) -> FileResponse:
        record = _media_or_404(database, media_id)
        path = Path(record.path)
        return FileResponse(path, media_type=mimetypes.guess_type(path.name)[0])

    @app.get("/api/media/{media_id}/thumbnail")
    def get_thumbnail(media_id: str, at: float | None = Query(default=None, ge=0)) -> FileResponse:
        record = _media_or_404(database, media_id)
        cache_name = (
            f"{media_id}-{at:.3f}.jpg"
            if record.media_type == "video" and at is not None
            else f"{media_id}.jpg"
        )
        cache_path = settings.data_dir / "cache" / "thumbnails" / cache_name
        if not cache_path.exists():
            source = Path(record.path)
            image = (
                load_image(source)
                if record.media_type == "image"
                else extract_video_frame(source, at or (record.duration or 0) / 2)
            )
            create_thumbnail(image, cache_path)
        return FileResponse(cache_path, media_type="image/jpeg")

    @app.get("/api/video/{media_id}/clip")
    def get_video_clip(
        media_id: str,
        start: float = Query(default=0, ge=0),
        end: float = Query(default=4, gt=0),
    ) -> FileResponse:
        record = _media_or_404(database, media_id)
        if record.media_type != "video":
            raise HTTPException(status_code=400, detail="该媒体不是视频")
        if end <= start:
            raise HTTPException(status_code=400, detail="end 必须大于 start")
        if record.duration is not None and start >= record.duration:
            raise HTTPException(status_code=416, detail="start 超出视频时长")
        duration = max(0.25, min(end - start, 30.0))
        cache_path = (
            settings.data_dir
            / "cache"
            / "clips"
            / f"{media_id}-{start:.3f}-{start + duration:.3f}.mp4"
        )
        if not cache_path.exists():
            command = [
                "ffmpeg",
                "-loglevel",
                "error",
                "-ss",
                f"{start:.3f}",
                "-i",
                record.path,
                "-t",
                f"{duration:.3f}",
                "-map",
                "0:v:0",
                "-map",
                "0:a?",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-c:a",
                "aac",
                "-movflags",
                "+faststart",
                "-y",
                str(cache_path),
            ]
            try:
                subprocess.run(command, capture_output=True, check=True, timeout=120)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
                cache_path.unlink(missing_ok=True)
                raise HTTPException(status_code=500, detail="视频片段生成失败") from error
        return FileResponse(cache_path, media_type="video/mp4")

    @app.post("/api/feedback/relevance", status_code=201)
    def feedback(request: FeedbackRequest) -> dict[str, str]:
        _media_or_404(database, request.media_id)
        database.add_feedback(request.query_id, request.media_id, request.relevance, request.note)
        return {"status": "recorded"}

    _mount_frontend(app)
    return app


async def _read_upload(file: UploadFile) -> Image.Image:
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="仅支持图片查询文件")
    payload = await file.read()
    if len(payload) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="图片不能超过 20 MB")
    try:
        with Image.open(io.BytesIO(payload)) as source:
            return source.convert("RGB")
    except OSError as error:
        raise HTTPException(status_code=400, detail="无法读取图片") from error


def _media_or_404(database: Database, media_id: str):
    record = database.get_media(media_id)
    if record is None or not Path(record.path).is_file():
        raise HTTPException(status_code=404, detail="媒体不存在")
    return record


def _with_urls(response: dict[str, object]) -> dict[str, object]:
    results = response.get("results", [])
    if isinstance(results, list):
        for result in results:
            if not isinstance(result, dict):
                continue
            media_id = result["media_id"]
            thumbnail_url = f"/api/media/{media_id}/thumbnail"
            if result.get("media_type") == "video" and result.get("thumbnail_sec") is not None:
                thumbnail_url += f"?at={result['thumbnail_sec']}"
            result["thumbnail_url"] = thumbnail_url
            result["media_url"] = f"/api/media/{media_id}"
            if result.get("media_type") == "video":
                result["clip_url"] = (
                    f"/api/video/{media_id}/clip?start={result.get('start_sec', 0)}"
                    f"&end={result.get('end_sec', 4)}"
                )
    return response


def _mount_frontend(app: FastAPI) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    web_dist = Path(os.getenv("SCENESEEK_WEB_DIST", repo_root / "web" / "dist")).resolve()
    if not web_dist.is_dir():
        return
    assets = web_dist / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def frontend(path: str) -> FileResponse:
        requested = (web_dist / path).resolve()
        if path and requested.is_relative_to(web_dist) and requested.is_file():
            return FileResponse(requested)
        return FileResponse(web_dist / "index.html")
