from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np
from PIL import Image


def l2_normalize(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(value))
    return value / norm if norm > 0 else value


class Encoder(ABC):
    name: str
    version: str
    dimension: int

    @abstractmethod
    def encode_text(self, text: str) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def encode_image(self, image: Image.Image, context: str = "") -> np.ndarray:
        raise NotImplementedError

    def encode_images(
        self,
        images: Sequence[Image.Image],
        contexts: Sequence[str] | None = None,
        *,
        batch_size: int | None = None,
    ) -> np.ndarray:
        del batch_size
        if contexts is not None and len(contexts) != len(images):
            raise ValueError("contexts 数量必须与 images 一致")
        vectors = [
            self.encode_image(image, context=contexts[index] if contexts is not None else "")
            for index, image in enumerate(images)
        ]
        if not vectors:
            return np.empty((0, self.dimension), dtype=np.float32)
        return np.stack(vectors).astype(np.float32, copy=False)
