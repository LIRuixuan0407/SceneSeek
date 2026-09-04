from __future__ import annotations

import uuid

import numpy as np
from PIL import Image

from sceneseek.config import Settings
from sceneseek.encoders import Encoder
from sceneseek.encoders.base import l2_normalize
from sceneseek.retrieval.index import VectorIndex
from sceneseek.temporal import aggregate_neighbors


class RetrievalService:
    def __init__(self, settings: Settings, encoder: Encoder, index: VectorIndex) -> None:
        self.settings = settings
        self.encoder = encoder
        self.index = index

    def search_text(
        self, query: str, *, limit: int = 24, media_type: str | None = None
    ) -> dict[str, object]:
        if not query.strip():
            raise ValueError("查询文本不能为空")
        vector = self.encoder.encode_text(query.strip())
        return self._search(vector, limit=limit, media_type=media_type, query=query.strip())

    def search_image(
        self, image: Image.Image, *, limit: int = 24, media_type: str | None = None
    ) -> dict[str, object]:
        vector = self.encoder.encode_image(image.convert("RGB"))
        return self._search(vector, limit=limit, media_type=media_type, query="image")

    def search_composed(
        self,
        text: str,
        image: Image.Image,
        *,
        text_weight: float = 0.55,
        limit: int = 24,
        media_type: str | None = None,
    ) -> dict[str, object]:
        weight = min(max(text_weight, 0.0), 1.0)
        vector = l2_normalize(
            self.encoder.encode_text(text) * weight
            + self.encoder.encode_image(image.convert("RGB")) * (1 - weight)
        )
        return self._search(vector, limit=limit, media_type=media_type, query=text)

    def _search(
        self,
        vector: np.ndarray,
        *,
        limit: int,
        media_type: str | None,
        query: str,
    ) -> dict[str, object]:
        if self.index.size == 0:
            raise RuntimeError("索引为空，请先扫描媒体目录并构建索引")
        limit = min(max(limit, 1), 100)
        neighbors = self.index.search(
            vector,
            max(self.settings.coarse_top_k, limit * 6),
            media_type=media_type,
        )
        results = aggregate_neighbors(
            neighbors,
            limit=limit,
            merge_gap=self.settings.window_stride_seconds,
        )
        return {
            "query_id": uuid.uuid4().hex,
            "query": query,
            "total": len(results),
            "results": [result.to_dict() for result in results],
        }
