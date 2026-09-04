from __future__ import annotations

import importlib
import sys

import sceneseek.encoders as encoders


def test_importing_service_app_does_not_create_encoder(monkeypatch) -> None:
    def fail_if_created(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("importing sceneseek.service.app must not create an encoder")

    monkeypatch.setattr(encoders, "create_encoder", fail_if_created)
    sys.modules.pop("sceneseek.service.app", None)

    module = importlib.import_module("sceneseek.service.app")

    assert callable(module.create_app)
    assert not hasattr(module, "app")
