# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for model snapshot download selection and verification."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from anlord.models.downloader import (
    RemoteFile,
    build_snapshot_plan,
    incomplete_snapshot_files,
    prefetch_model_snapshot,
    reject_unsupported_weight_layout,
    select_download_patterns,
)


def test_safetensors_are_preferred_over_duplicate_bin_weights():
    patterns = select_download_patterns(
        [
            "config.json",
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
            "pytorch_model.bin",
        ]
    )

    assert "*.safetensors" in patterns
    assert "*.bin" not in patterns
    assert "*.json" in patterns


def test_bin_weights_are_supported_when_safetensors_are_absent():
    patterns = select_download_patterns(["config.json", "pytorch_model.bin"])

    assert "*.bin" in patterns
    assert "*.safetensors" not in patterns


def test_unknown_weight_layout_falls_back_to_full_snapshot():
    assert select_download_patterns(["config.json", "custom.weights"]) is None


def test_incomplete_zero_byte_and_lfs_pointer_weights_are_detected(tmp_path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"")
    (tmp_path / "tiny.safetensors").write_text(
        "version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 10\n",
        encoding="utf-8",
    )
    (tmp_path / "ok.safetensors").write_bytes(b"x" * 100)

    missing = incomplete_snapshot_files(
        tmp_path,
        [
            RemoteFile("config.json", 2),
            RemoteFile("model.safetensors", 100),
            RemoteFile("tiny.safetensors", 100),
            RemoteFile("ok.safetensors", 100),
            RemoteFile("absent.safetensors", 50),
        ],
    )
    names = {item.name for item in missing}
    assert names == {"model.safetensors", "tiny.safetensors", "absent.safetensors"}


def test_snapshot_plan_uses_remote_sizes():
    plan = build_snapshot_plan(
        [
            RemoteFile("config.json", 10),
            RemoteFile("model.safetensors", 5 * 1024**3),
            RemoteFile("model.onnx", 99),
        ]
    )

    assert plan.allow_patterns is not None
    assert plan.expected_bytes == 10 + 5 * 1024**3
    assert [item.name for item in plan.weight_files] == ["model.safetensors"]


def test_gguf_only_repo_is_rejected_with_base_model_hint():
    files = [
        RemoteFile("Ornith-1.5-9B-Q4_K_M.gguf", 5_000_000_000),
        RemoteFile("Ornith-1.5-9B-Q5_K_M.gguf", 6_000_000_000),
        RemoteFile("README.md", 1000),
    ]
    plan = build_snapshot_plan(files)
    with pytest.raises(RuntimeError, match="ornith-ai/Ornith-1.5-9B") as error:
        reject_unsupported_weight_layout("ornith-ai/Ornith-1.5-9B-GGUF", files, plan)

    assert "GGUF" in str(error.value)
    assert "Q4_K_M" in str(error.value)


def test_prefetch_passes_ephemeral_token_and_progress_class(tmp_path, monkeypatch):
    captured = {}
    hub_module = ModuleType("huggingface_hub")

    class FakeApi:
        def __init__(self, token=None):
            captured["api_token"] = token

        def model_info(self, repo_id, revision=None, files_metadata=False):
            captured["listed"] = (repo_id, revision, files_metadata)
            return SimpleNamespace(
                siblings=[
                    SimpleNamespace(rfilename="config.json", size=2),
                    SimpleNamespace(rfilename="model.safetensors", size=100),
                ]
            )

    def fake_snapshot_download(**kwargs):
        captured.setdefault("snapshots", []).append(kwargs)
        snapshot = tmp_path / "snapshot"
        snapshot.mkdir(exist_ok=True)
        (snapshot / "config.json").write_text("{}", encoding="utf-8")
        (snapshot / "model.safetensors").write_bytes(b"x" * 100)
        return str(snapshot)

    hub_module.HfApi = FakeApi
    hub_module.snapshot_download = fake_snapshot_download
    hub_errors = ModuleType("huggingface_hub.errors")
    hub_errors.GatedRepoError = type("GatedRepoError", (Exception,), {})
    hub_errors.HfHubHTTPError = type("HfHubHTTPError", (Exception,), {})
    tqdm_package = ModuleType("tqdm")
    tqdm_auto = ModuleType("tqdm.auto")

    class FakeTqdm:
        def __init__(self, *args, **kwargs):
            pass

        def update(self, n=1):
            return None

        def close(self):
            return None

    tqdm_auto.tqdm = FakeTqdm
    tqdm_package.auto = tqdm_auto
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub_module)
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", hub_errors)
    monkeypatch.setitem(sys.modules, "tqdm", tqdm_package)
    monkeypatch.setitem(sys.modules, "tqdm.auto", tqdm_auto)
    monkeypatch.setenv("HF_TOKEN", "hf_ephemeral")

    result = prefetch_model_snapshot(
        "example/model",
        cache_dir=tmp_path / "cache",
        revision="abc123",
    )

    assert result.endswith("snapshot")
    assert captured["api_token"] == "hf_ephemeral"
    assert captured["snapshots"][0]["token"] == "hf_ephemeral"
    assert captured["snapshots"][0]["tqdm_class"].__name__ == "VisibleTqdm"
    assert captured["snapshots"][0]["allow_patterns"][-1] == "*.safetensors"


def test_prefetch_skips_download_when_complete_cache_exists(tmp_path, monkeypatch):
    captured = {}
    hub_module = ModuleType("huggingface_hub")
    repo = tmp_path / "cache" / "hub" / "models--example--model" / "snapshots" / "abc123"
    repo.mkdir(parents=True)
    (repo / "config.json").write_text("{}", encoding="utf-8")
    (repo / "model.safetensors").write_bytes(b"x" * 100)

    class FakeApi:
        def __init__(self, token=None):
            pass

        def model_info(self, repo_id, revision=None, files_metadata=False):
            return SimpleNamespace(
                siblings=[
                    SimpleNamespace(rfilename="config.json", size=2),
                    SimpleNamespace(rfilename="model.safetensors", size=100),
                ]
            )

    def fake_snapshot_download(**kwargs):
        captured["called"] = True
        return "should-not-run"

    hub_module.HfApi = FakeApi
    hub_module.snapshot_download = fake_snapshot_download
    hub_errors = ModuleType("huggingface_hub.errors")
    hub_errors.GatedRepoError = type("GatedRepoError", (Exception,), {})
    hub_errors.HfHubHTTPError = type("HfHubHTTPError", (Exception,), {})
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub_module)
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", hub_errors)

    result = prefetch_model_snapshot("example/model", cache_dir=tmp_path / "cache")

    assert result == str(repo)
    assert "called" not in captured
