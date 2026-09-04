from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from sceneseek.encoders.base import Encoder, l2_normalize


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
        inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self._torch.inference_mode():
            vector = self.model.get_text_features(**inputs)[0]
        return l2_normalize(vector.float().cpu().numpy())

    def encode_image(self, image: Image.Image, context: str = "") -> np.ndarray:
        inputs = self.processor(images=[image], return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self._torch.inference_mode():
            vector = self.model.get_image_features(**inputs)[0]
        return l2_normalize(vector.float().cpu().numpy())
