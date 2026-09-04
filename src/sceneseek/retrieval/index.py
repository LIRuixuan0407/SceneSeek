from __future__ import annotations

import importlib.util
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sceneseek.encoders.base import l2_normalize
from sceneseek.storage import Database


@dataclass(slots=True)
class IndexedItem:
    item_id: str
    media_id: str
    kind: str
    path: str
    media_type: str
    duration: float | None
    width: int | None
    height: int | None
    start_sec: float | None
    end_sec: float | None


@dataclass(slots=True)
class Neighbor:
    item: IndexedItem
    score: float


class VectorIndex:
    def __init__(self, path: Path, backend: str = "auto") -> None:
        self.path = Path(path)
        self.requested_backend = backend
        self.backend = "numpy-flat"
        self.items: list[IndexedItem] = []
        self.vectors = np.empty((0, 0), dtype=np.float32)
        self._faiss: Any | None = None
        if self.path.exists():
            self.load()

    @property
    def size(self) -> int:
        return len(self.items)

    @property
    def dimension(self) -> int:
        return int(self.vectors.shape[1]) if self.vectors.ndim == 2 and self.vectors.size else 0

    def rebuild(self, database: Database, version: str) -> int:
        records = list(database.iter_embeddings(version))
        items = [
            IndexedItem(
                item_id=str(record["item_id"]),
                media_id=str(record["media_id"]),
                kind=str(record["kind"]),
                path=str(record["path"]),
                media_type=str(record["media_type"]),
                duration=float(record["duration"]) if record["duration"] is not None else None,
                width=int(record["width"]) if record["width"] is not None else None,
                height=int(record["height"]) if record["height"] is not None else None,
                start_sec=(float(record["start_sec"]) if record["start_sec"] is not None else None),
                end_sec=float(record["end_sec"]) if record["end_sec"] is not None else None,
            )
            for record in records
        ]
        vectors = (
            np.stack([np.asarray(record["vector"], dtype=np.float32) for record in records])
            if records
            else np.empty((0, 0), dtype=np.float32)
        )
        self.set_data(items, vectors)
        self.save()
        return self.size

    def set_data(self, items: list[IndexedItem], vectors: np.ndarray) -> None:
        matrix = np.asarray(vectors, dtype=np.float32)
        if items and (matrix.ndim != 2 or matrix.shape[0] != len(items)):
            raise ValueError("向量数量与索引元数据不一致")
        if matrix.size:
            matrix = np.stack([l2_normalize(row) for row in matrix])
        self.items = items
        self.vectors = matrix
        self._configure_backend()

    def search(
        self,
        query: np.ndarray,
        top_k: int,
        *,
        media_type: str | None = None,
    ) -> list[Neighbor]:
        if not self.items or top_k <= 0:
            return []
        vector = l2_normalize(query).astype(np.float32)
        if vector.size != self.dimension:
            raise ValueError(f"查询向量维度 {vector.size} 与索引维度 {self.dimension} 不一致")

        request_k = min(len(self.items), max(top_k * 4, top_k))
        if self._faiss is not None:
            scores, indices = self._faiss.search(vector.reshape(1, -1), request_k)
            pairs = zip(indices[0].tolist(), scores[0].tolist(), strict=False)
        else:
            similarities = self.vectors @ vector
            candidate_indices = np.argpartition(
                -similarities, min(request_k, len(similarities)) - 1
            )[:request_k]
            candidate_indices = candidate_indices[
                np.argsort(-similarities[candidate_indices], kind="stable")
            ]
            pairs = ((int(index), float(similarities[index])) for index in candidate_indices)

        neighbors: list[Neighbor] = []
        for index, score in pairs:
            if index < 0:
                continue
            item = self.items[index]
            if media_type and item.media_type != media_type:
                continue
            neighbors.append(Neighbor(item=item, score=float(score)))
            if len(neighbors) >= top_k:
                break
        return neighbors

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        metadata = json.dumps([asdict(item) for item in self.items], ensure_ascii=False)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, vectors=self.vectors, metadata=np.array(metadata))
        temporary.replace(self.path)

    def load(self) -> None:
        with np.load(self.path, allow_pickle=False) as archive:
            vectors = np.asarray(archive["vectors"], dtype=np.float32)
            metadata = json.loads(str(archive["metadata"].item()))
        self.set_data([IndexedItem(**item) for item in metadata], vectors)

    def _configure_backend(self) -> None:
        self._faiss = None
        wants_faiss = self.requested_backend in {"auto", "faiss", "flat", "hnsw"}
        has_faiss = importlib.util.find_spec("faiss") is not None
        if not wants_faiss or not has_faiss or not self.vectors.size:
            self.backend = "numpy-flat"
            return
        import faiss

        use_hnsw = self.requested_backend == "hnsw" or (
            self.requested_backend == "auto" and len(self.items) >= 50_000
        )
        if use_hnsw:
            index = faiss.IndexHNSWFlat(self.dimension, 32, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efSearch = 64
            self.backend = "faiss-hnsw"
        else:
            index = faiss.IndexFlatIP(self.dimension)
            self.backend = "faiss-flat"
        index.add(np.ascontiguousarray(self.vectors))
        self._faiss = index
