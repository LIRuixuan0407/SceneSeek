from __future__ import annotations

from abc import ABC, abstractmethod

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
