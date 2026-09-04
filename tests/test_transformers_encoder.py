from __future__ import annotations

from types import SimpleNamespace

from sceneseek.encoders.transformers import _feature_tensor


def test_feature_tensor_accepts_legacy_tensor_like_output() -> None:
    sentinel = object()
    assert _feature_tensor(sentinel) is sentinel


def test_feature_tensor_extracts_pooler_output_from_model_output() -> None:
    sentinel = object()
    output = SimpleNamespace(pooler_output=sentinel)
    assert _feature_tensor(output) is sentinel
