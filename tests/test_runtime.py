# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for runtime dependency and cache compatibility helpers."""

from __future__ import annotations

import os
import sys
from types import ModuleType, SimpleNamespace

from anlord.utils import runtime
from anlord.utils.runtime import configure_huggingface_environment


def test_configure_huggingface_environment_updates_imported_constants(tmp_path, monkeypatch):
    constants = SimpleNamespace(HF_HOME="old", HF_HUB_CACHE="old", HUGGINGFACE_HUB_CACHE="old")
    datasets_config = SimpleNamespace(HF_DATASETS_CACHE="old")
    monkeypatch.setitem(sys.modules, "huggingface_hub.constants", constants)
    monkeypatch.setitem(sys.modules, "datasets.config", datasets_config)

    cache = configure_huggingface_environment(tmp_path / "cache")

    assert os.environ["HF_HOME"] == str(cache)
    assert constants.HF_HUB_CACHE == str(cache / "hub")
    assert constants.HUGGINGFACE_HUB_CACHE == str(cache / "hub")
    assert datasets_config.HF_DATASETS_CACHE == str(cache / "datasets")


def test_broken_optional_torchaudio_is_disabled(monkeypatch):
    transformers = ModuleType("transformers")
    transformers_utils = ModuleType("transformers.utils")
    import_utils = ModuleType("transformers.utils.import_utils")

    def available():
        return True

    available.cache_clear = lambda: None
    import_utils.is_torchaudio_available = available
    transformers_utils.import_utils = import_utils
    transformers_utils.is_torchaudio_available = available
    transformers.utils = transformers_utils
    transformers.is_torchaudio_available = available

    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "transformers.utils", transformers_utils)
    monkeypatch.setitem(sys.modules, "transformers.utils.import_utils", import_utils)
    monkeypatch.setattr(runtime, "_OPTIONAL_DEPENDENCIES_PREPARED", False)

    real_import_module = runtime.importlib.import_module

    def import_with_broken_torchaudio(name, *args, **kwargs):
        if name == "torchaudio":
            raise OSError("libtorchaudio.pyd cannot be loaded")
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(runtime.importlib, "import_module", import_with_broken_torchaudio)
    runtime.prepare_optional_ml_dependencies()

    assert import_utils.is_torchaudio_available() is False
    assert transformers_utils.is_torchaudio_available() is False


def test_free_torch_memory_does_not_raise():
    from anlord.utils.runtime import free_torch_memory

    free_torch_memory()


def test_free_torch_memory_does_not_initialize_cuda(monkeypatch):
    import torch

    from anlord.utils import runtime

    calls = {"available": 0, "empty": 0}

    def is_available():
        calls["available"] += 1
        return True

    def empty_cache():
        calls["empty"] += 1

    fake_cuda = SimpleNamespace(
        is_initialized=lambda: False,
        is_available=is_available,
        empty_cache=empty_cache,
    )
    monkeypatch.setattr(torch, "cuda", fake_cuda)
    runtime.free_torch_memory()

    assert calls["available"] == 0
    assert calls["empty"] == 0
