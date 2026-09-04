from __future__ import annotations

import time
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from sceneseek.config import Settings
from sceneseek.service.app import create_app


def wait_for_job(client: TestClient) -> dict[str, object]:
    for _ in range(100):
        status = client.get("/api/index/status").json()
        if status["state"] != "running":
            return status
        time.sleep(0.02)
    raise AssertionError("background job did not complete")


def test_scan_build_and_search_workflow(tmp_path: Path) -> None:
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    Image.new("RGB", (80, 60), "red").save(media_dir / "red-sunset.jpg")
    Image.new("RGB", (80, 60), "blue").save(media_dir / "blue-ocean.jpg")
    settings = Settings(
        data_dir=tmp_path / "state",
        encoder="lite",
        index_backend="numpy",
        allowed_roots=(tmp_path,),
        cors_origins=("http://test",),
    )

    with TestClient(create_app(settings)) as client:
        assert client.post("/api/index/scan", json={"path": str(media_dir)}).status_code == 202
        assert wait_for_job(client)["state"] == "complete"
        assert client.post("/api/index/build", json={"rebuild": False}).status_code == 202
        status = wait_for_job(client)
        assert status["state"] == "complete"
        assert status["index"]["size"] == 2  # type: ignore[index]

        response = client.post("/api/search/text", json={"query": "red sunset", "limit": 2})
        assert response.status_code == 200
        payload = response.json()
        assert payload["results"][0]["path"].endswith("red-sunset.jpg")
        assert payload["results"][0]["thumbnail_url"].startswith("/api/media/")

        with client.app.state.database.connect() as connection:
            query_row = connection.execute(
                "SELECT query_text, query_type, model_version FROM queries WHERE query_id = ?",
                (payload["query_id"],),
            ).fetchone()
            impressions = connection.execute(
                "SELECT rank, media_id, score FROM impressions WHERE query_id = ? ORDER BY rank",
                (payload["query_id"],),
            ).fetchall()
        assert query_row is not None
        assert query_row["query_text"] == "red sunset"
        assert query_row["query_type"] == "text"
        assert len(impressions) == len(payload["results"])
