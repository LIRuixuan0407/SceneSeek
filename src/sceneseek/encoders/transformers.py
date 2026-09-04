from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from PIL import Image

from sceneseek.encoders.base import Encoder, l2_normalize


def _feature_tensor(output: Any) -> Any:
    """Return the projected embedding tensor across Transformers API versions."""
    pooled = getattr(output, "pooler_output", None)
    return pooled if pooled is not None else output


class TransformersEncoder(Encoder):
    name = "transformers"

    def __init__(self, model_id: str, device: str = "auto") -> None:
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except ImportError as error:
            raise RuntimeError("请先安装 `pip install -e '.[ml]'` 以启用 CLIP/SigLIP") from error

        self._torch = torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model_id = model_id
        self.version = f"transformers:{model_id}"
        self.processor: Any = AutoProcessor.from_pretrained(model_id)
        self.model: Any = AutoModel.from_pretrained(model_id).to(device).eval()
        projection = getattr(self.model.config, "projection_dim", None)
        self.dimension = int(projection or 512)

    def encode_text(self, text: str) -> np.ndarray:
        return self.encode_texts([text], batch_size=1)[0]

    def encode_texts(
        self,
        texts: Sequence[str],
        *,
        batch_size: int | None = None,
    ) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        size = max(1, batch_size or len(texts))
        chunks: list[np.ndarray] = []
        for start in range(0, len(texts), size):
            batch = list(texts[start : start + size])
            inputs = self.processor(text=batch, return_tensors="pt", padding=True)
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with self._torch.inference_mode():
                vectors = _feature_tensor(self.model.get_text_features(**inputs)).float().cpu().numpy()
            normalized = np.stack([l2_normalize(vector) for vector in vectors])
            chunks.append(normalized.astype(np.float32, copy=False))
        return np.concatenate(chunks, axis=0)

    def encode_image(self, image: Image.Image, context: str = "") -> np.ndarray:
        del context
        return self.encode_images([image], batch_size=1)[0]

    def encode_images(
        self,
        images: Sequence[Image.Image],
        contexts: Sequence[str] | None = None,
        *,
        batch_size: int | None = None,
    ) -> np.ndarray:
        del contexts
        if not images:
            return np.empty((0, self.dimension), dtype=np.float32)
        size = max(1, batch_size or len(images))
        chunks: list[np.ndarray] = []
        for start in range(0, len(images), size):
            batch = list(images[start : start + size])
            inputs = self.processor(images=batch, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with self._torch.inference_mode():
                vectors = _feature_tensor(self.model.get_image_features(**inputs)).float().cpu().numpy()
            normalized = np.stack([l2_normalize(vector) for vector in vectors])
            chunks.append(normalized.astype(np.float32, copy=False))
        return np.concatenate(chunks, axis=0)
