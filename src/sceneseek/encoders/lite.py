from __future__ import annotations

import hashlib
import re
from collections import Counter

import numpy as np
from PIL import Image

from sceneseek.encoders.base import Encoder, l2_normalize

COLOR_TERMS = {
    0: ("red", "crimson", "scarlet", "红", "红色"),
    1: ("orange", "橙", "橙色"),
    2: ("yellow", "gold", "黄色", "金色", "落日", "sunset"),
    3: ("green", "绿色", "草地", "forest", "树林"),
    4: ("cyan", "青色"),
    5: ("blue", "蓝", "蓝色", "天空", "sky", "海", "ocean"),
    6: ("purple", "violet", "紫", "紫色"),
    7: ("pink", "粉", "粉色"),
    8: ("black", "dark", "night", "黑", "夜", "夜晚", "雨夜"),
    9: ("white", "bright", "snow", "白", "雪", "明亮"),
    10: ("gray", "grey", "灰", "灰色"),
    11: ("brown", "棕", "棕色"),
}


class LiteEncoder(Encoder):
    """Zero-download fallback using color concepts and hashed text context.

    This keeps the product runnable on CPU-only machines. Install the ``ml`` extra
    for the intended CLIP/SigLIP baseline.
    """

    name = "lite"
    version = "lite-color-text-v1"
    dimension = 256

    def encode_text(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype=np.float32)
        lowered = text.casefold()
        for index, terms in COLOR_TERMS.items():
            if any(term in lowered for term in terms):
                vector[index] += 2.5

        tokens = self._tokens(lowered)
        for token, count in Counter(tokens).items():
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = 16 + int.from_bytes(digest, "little") % (self.dimension - 16)
            sign = 1.0 if digest[0] % 2 == 0 else -1.0
            vector[bucket] += sign * min(count, 3)
        return l2_normalize(vector)

    def encode_image(self, image: Image.Image, context: str = "") -> np.ndarray:
        pixels = np.asarray(image.convert("RGB").resize((96, 96)), dtype=np.float32) / 255.0
        hsv = np.asarray(image.convert("HSV").resize((96, 96)), dtype=np.float32)
        hue = hsv[..., 0] / 255.0 * 360.0
        saturation = hsv[..., 1] / 255.0
        value = hsv[..., 2] / 255.0

        vector = np.zeros(self.dimension, dtype=np.float32)
        chromatic = saturation > 0.22
        hue_centers = [0, 30, 58, 118, 180, 220, 278, 330]
        for index, center in enumerate(hue_centers):
            distance = np.minimum(abs(hue - center), 360 - abs(hue - center))
            vector[index] = float(np.mean((distance < 24) & chromatic))
        vector[8] = float(np.mean(value < 0.22))
        vector[9] = float(np.mean((value > 0.82) & (saturation < 0.22)))
        vector[10] = float(np.mean((saturation < 0.18) & (value >= 0.22) & (value <= 0.82)))
        vector[11] = float(np.mean((hue > 12) & (hue < 42) & chromatic & (value < 0.62)))
        vector[12:15] = pixels.mean(axis=(0, 1))
        vector[15] = float(pixels.std())

        visual = l2_normalize(vector)
        if context:
            context_vector = self.encode_text(context)
            visual = l2_normalize(visual * 0.78 + context_vector * 0.22)
        return visual

    @staticmethod
    def _tokens(text: str) -> list[str]:
        latin = re.findall(r"[a-z0-9]+", text)
        chinese = re.findall(r"[\u4e00-\u9fff]", text)
        chinese += ["".join(chinese[i : i + 2]) for i in range(max(0, len(chinese) - 1))]
        return latin + chinese
