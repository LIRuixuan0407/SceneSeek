from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from sceneseek.domain import ClipRecord, MediaRecord

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS media (
    media_id TEXT PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    media_type TEXT NOT NULL CHECK(media_type IN ('image', 'video')),
    duration REAL,
    fps REAL,
    width INTEGER,
    height INTEGER,
    mtime REAL NOT NULL,
    size_bytes INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    index_version TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_media_hash ON media(content_hash);
CREATE INDEX IF NOT EXISTS idx_media_type ON media(media_type);

CREATE TABLE IF NOT EXISTS clips (
    clip_id TEXT PRIMARY KEY,
    media_id TEXT NOT NULL REFERENCES media(media_id) ON DELETE CASCADE,
    start_sec REAL NOT NULL,
    end_sec REAL NOT NULL,
    thumbnail_sec REAL NOT NULL,
    sampled_frame_ts TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_clips_media ON clips(media_id, start_sec);

CREATE TABLE IF NOT EXISTS embeddings (
    item_id TEXT PRIMARY KEY,
    media_id TEXT NOT NULL REFERENCES media(media_id) ON DELETE CASCADE,
    kind TEXT NOT NULL CHECK(kind IN ('image', 'clip')),
    vector BLOB NOT NULL,
    dim INTEGER NOT NULL,
    model_version TEXT NOT NULL,
    start_sec REAL,
    end_sec REAL
);

CREATE INDEX IF NOT EXISTS idx_embeddings_media ON embeddings(media_id);

CREATE TABLE IF NOT EXISTS feedback (
    feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_id TEXT NOT NULL,
    media_id TEXT NOT NULL,
    relevance INTEGER NOT NULL CHECK(relevance BETWEEN 0 AND 3),
    note TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""


class Database:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.executescript(SCHEMA)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def get_media_by_path(self, path: str) -> MediaRecord | None:
        with self.connect() as connection:
            row = connection.execute("SELECT * FROM media WHERE path = ?", (path,)).fetchone()
        return self._media_from_row(row) if row else None

    def get_media(self, media_id: str) -> MediaRecord | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM media WHERE media_id = ?", (media_id,)
            ).fetchone()
        return self._media_from_row(row) if row else None

    def find_by_hash(
        self, content_hash: str, exclude_path: str | None = None
    ) -> MediaRecord | None:
        sql = "SELECT * FROM media WHERE content_hash = ?"
        params: tuple[object, ...] = (content_hash,)
        if exclude_path is not None:
            sql += " AND path != ?"
            params = (content_hash, exclude_path)
        sql += " LIMIT 1"
        with self.connect() as connection:
            row = connection.execute(sql, params).fetchone()
        return self._media_from_row(row) if row else None

    def upsert_media(self, record: MediaRecord) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO media (
                    media_id, path, media_type, duration, fps, width, height,
                    mtime, size_bytes, content_hash, index_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    media_type=excluded.media_type,
                    duration=excluded.duration,
                    fps=excluded.fps,
                    width=excluded.width,
                    height=excluded.height,
                    mtime=excluded.mtime,
                    size_bytes=excluded.size_bytes,
                    content_hash=excluded.content_hash,
                    index_version=excluded.index_version,
                    updated_at=CURRENT_TIMESTAMP
                """,
                (
                    record.media_id,
                    record.path,
                    record.media_type,
                    record.duration,
                    record.fps,
                    record.width,
                    record.height,
                    record.mtime,
                    record.size_bytes,
                    record.content_hash,
                    record.index_version,
                ),
            )

    def replace_clips(self, media_id: str, clips: Iterable[ClipRecord]) -> None:
        with self.connect() as connection:
            connection.execute("DELETE FROM clips WHERE media_id = ?", (media_id,))
            connection.executemany(
                """
                INSERT INTO clips (
                    clip_id, media_id, start_sec, end_sec, thumbnail_sec, sampled_frame_ts
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        clip.clip_id,
                        clip.media_id,
                        clip.start_sec,
                        clip.end_sec,
                        clip.thumbnail_sec,
                        json.dumps(clip.sampled_frame_ts),
                    )
                    for clip in clips
                ],
            )

    def list_clips(self, media_id: str) -> list[ClipRecord]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM clips WHERE media_id = ? ORDER BY start_sec", (media_id,)
            ).fetchall()
        return [
            ClipRecord(
                clip_id=row["clip_id"],
                media_id=row["media_id"],
                start_sec=row["start_sec"],
                end_sec=row["end_sec"],
                thumbnail_sec=row["thumbnail_sec"],
                sampled_frame_ts=tuple(json.loads(row["sampled_frame_ts"])),
            )
            for row in rows
        ]

    def list_media(self, *, needs_version: str | None = None) -> list[MediaRecord]:
        sql = "SELECT * FROM media"
        params: tuple[object, ...] = ()
        if needs_version is not None:
            sql += " WHERE index_version IS NULL OR index_version != ?"
            params = (needs_version,)
        sql += " ORDER BY path"
        with self.connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [self._media_from_row(row) for row in rows]

    def delete_missing_under(self, root: Path, present_paths: set[str]) -> list[str]:
        prefix = str(root.resolve())
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT media_id, path FROM media WHERE path = ? OR path LIKE ?",
                (prefix, f"{prefix.rstrip('/')}/%"),
            ).fetchall()
            missing = [row["media_id"] for row in rows if row["path"] not in present_paths]
            if missing:
                connection.executemany(
                    "DELETE FROM media WHERE media_id = ?", [(media_id,) for media_id in missing]
                )
        return missing

    def invalidate_media(self, media_id: str) -> None:
        with self.connect() as connection:
            connection.execute("DELETE FROM embeddings WHERE media_id = ?", (media_id,))
            connection.execute(
                "UPDATE media SET index_version = NULL WHERE media_id = ?", (media_id,)
            )

    def put_embedding(
        self,
        *,
        item_id: str,
        media_id: str,
        kind: str,
        vector: np.ndarray,
        model_version: str,
        start_sec: float | None = None,
        end_sec: float | None = None,
    ) -> None:
        normalized = np.asarray(vector, dtype=np.float32)
        with self.connect() as connection:
            connection.execute(
                """
                INSERT INTO embeddings (
                    item_id, media_id, kind, vector, dim, model_version, start_sec, end_sec
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(item_id) DO UPDATE SET
                    vector=excluded.vector,
                    dim=excluded.dim,
                    model_version=excluded.model_version,
                    start_sec=excluded.start_sec,
                    end_sec=excluded.end_sec
                """,
                (
                    item_id,
                    media_id,
                    kind,
                    normalized.tobytes(),
                    normalized.size,
                    model_version,
                    start_sec,
                    end_sec,
                ),
            )

    def mark_indexed(self, media_id: str, version: str) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE media
                SET index_version = ?, updated_at=CURRENT_TIMESTAMP
                WHERE media_id = ?
                """,
                (version, media_id),
            )

    def iter_embeddings(self, version: str) -> Iterator[dict[str, object]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT e.*, m.path, m.media_type, m.duration, m.width, m.height
                FROM embeddings e
                JOIN media m ON m.media_id = e.media_id
                WHERE e.model_version = ? AND m.index_version = ?
                ORDER BY e.item_id
                """,
                (version, version),
            ).fetchall()
        for row in rows:
            item = dict(row)
            item["vector"] = np.frombuffer(row["vector"], dtype=np.float32).copy()
            yield item

    def counts(self) -> dict[str, int]:
        with self.connect() as connection:
            media = connection.execute(
                "SELECT media_type, COUNT(*) AS count FROM media GROUP BY media_type"
            ).fetchall()
            clip_count = connection.execute("SELECT COUNT(*) FROM clips").fetchone()[0]
            indexed = connection.execute(
                "SELECT COUNT(DISTINCT media_id) FROM embeddings"
            ).fetchone()[0]
        result = {"images": 0, "videos": 0, "clips": clip_count, "indexed_media": indexed}
        for row in media:
            result[f"{row['media_type']}s"] = row["count"]
        return result

    def add_feedback(self, query_id: str, media_id: str, relevance: int, note: str | None) -> None:
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO feedback (query_id, media_id, relevance, note) VALUES (?, ?, ?, ?)",
                (query_id, media_id, relevance, note),
            )

    @staticmethod
    def _media_from_row(row: sqlite3.Row) -> MediaRecord:
        return MediaRecord(
            media_id=row["media_id"],
            path=row["path"],
            media_type=row["media_type"],
            duration=row["duration"],
            fps=row["fps"],
            width=row["width"],
            height=row["height"],
            mtime=row["mtime"],
            size_bytes=row["size_bytes"],
            content_hash=row["content_hash"],
            index_version=row["index_version"],
        )
