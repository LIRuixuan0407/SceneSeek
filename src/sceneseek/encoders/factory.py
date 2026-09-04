from __future__ import annotations

import importlib.util

from sceneseek.config import Settings
from sceneseek.encoders.base import Encoder
from sceneseek.encoders.lite import LiteEncoder
from sceneseek.encoders.transformers import TransformersEncoder


def create_encoder(settings: Settings) -> Encoder:
    backend = settings.encoder.casefold()
    if backend == "lite":
        return LiteEncoder()
    if backend in {"clip", "siglip", "transformers"}:
        return TransformersEncoder(settings.model_id, settings.device)
    if backend == "auto":
        has_ml = (
            importlib.util.find_spec("torch") is not None
            and importlib.util.find_spec("transformers") is not None
        )
        return TransformersEncoder(settings.model_id, settings.device) if has_ml else LiteEncoder()
    raise ValueError(f"不支持的 encoder backend: {settings.encoder}")
