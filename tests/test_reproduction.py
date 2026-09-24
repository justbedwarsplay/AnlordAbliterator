# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for reproduction support (bundle generation, hash verification, restore)."""

import hashlib
from dataclasses import fields as dataclass_fields
from pathlib import Path


from anlord.config import Settings
from anlord.native.config import NativeConfig, QuantizationMethod
from anlord.native.reproduce import (
    collect_model_hashes,
    create_reproduction_folder,
    generate_sha256sums,
    load_reproduction_information,
    settings_from_reproduction,
    verify_model_hashes,
    REPRODUCTION_SCHEMA_VERSION,
)


class _FakeTrial:
    """Duck-typed FrozenTrial with the user_attrs the bundle code reads."""

    def __init__(self):
        self.user_attrs = {
            "index": 7,
            "direction_index": 2.5,
            "parameters": {
                "attn.o_proj": {
                    "max_weight": 1.1,
                    "max_weight_position": 3.0,
                    "min_weight": 0.2,
                    "min_weight_distance": 2.0,
                }
            },
            "scores": [
                {
                    "name": "Refusals",
                    "score": {"value": 0.1, "rich_display": "[bold]1[/]/10", "md_display": "1/10"},
                    "baseline": {"value": 0.9, "rich_display": "[bold]9[/]/10", "md_display": "9/10"},
                },
                {
                    "name": "KL divergence",
                    "score": {"value": 0.02, "rich_display": "[bold]0.0200[/]", "md_display": "0.0200"},
                    "baseline": {"value": 0, "rich_display": "[bold]0[/] [italic](by definition)[/]", "md_display": "0 *(by definition)*"},
                },
            ],
            # Legacy summary metrics the pipeline reports.
            "base_refusals": 9,
            "refusals": 1,
            "n_bad_prompts": 10,
            "kl_divergence": 0.0212,
        }


def _make_native_config(tmp_path: Path) -> NativeConfig:
    return NativeConfig(
        model=str(tmp_path / "some_model"),
        dtypes=["float32"],
        quantization=QuantizationMethod.NONE,
        device_map="cpu",
        batch_size=2,
        offload_outputs_to_cpu=False,
        seed=42,
        export_strategy="merge",
        study_checkpoint_dir=str(tmp_path / "checkpoints"),
    )


def _write_weight_files(model_dir: Path) -> dict[str, str]:
    model_dir.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"):
        content = name.encode() * 1000
        (model_dir / name).write_bytes(content)
        hashes[name] = hashlib.sha256(content).hexdigest()
    return hashes


def test_collect_and_verify_model_hashes(tmp_path):
    model_dir = tmp_path / "model"
    hashes = _write_weight_files(model_dir)
    assert len(hashes) == 2
    assert collect_model_hashes(model_dir) == hashes

    report = verify_model_hashes(model_dir, hashes)
    assert all(status == "match" for status in report.values())

    # Tamper with one file.
    (model_dir / "model-00001-of-00002.safetensors").write_bytes(b"tampered")
    report = verify_model_hashes(model_dir, hashes)
    assert report["model-00001-of-00002.safetensors"] == "mismatch"
    assert report["model-00002-of-00002.safetensors"] == "match"

    report = verify_model_hashes(model_dir, {"missing.safetensors": "0" * 64})
    assert report["missing.safetensors"] == "not_found"


def test_generate_sha256sums_format():
    text = generate_sha256sums({"b.safetensors": "bb", "a.safetensors": "aa"})
    lines = text.strip().splitlines()
    assert lines == ["aa *a.safetensors", "bb *b.safetensors"]


def test_create_and_reload_reproduction_bundle(tmp_path):
    model_dir = tmp_path / "model"
    hashes = _write_weight_files(model_dir)
    native_cfg = _make_native_config(tmp_path)
    settings = Settings(model=native_cfg.model, output_dir=tmp_path / "out",
                        cache_dir=tmp_path / "cache")

    bundle_dir = create_reproduction_folder(
        model_dir,
        native_cfg,
        settings,
        checkpoint_path=None,
        trial=_FakeTrial(),
        model_hashes=hashes,
        include_system_information=True,
    )

    assert bundle_dir == model_dir / "reproduce"
    for name in ("reproduce.json", "SHA256SUMS", "requirements.txt", "config.json", "README.md"):
        assert (bundle_dir / name).is_file(), name

    info = load_reproduction_information(str(bundle_dir / "reproduce.json"))
    assert info["version"] == REPRODUCTION_SCHEMA_VERSION
    assert info["parameters"]["direction_index"] == 2.5
    assert info["hashes"] == hashes
    assert info["system"]["python"]["version"]
    assert info["native_config"]["response_prefix"] is None
    assert info["native_config"]["study_checkpoint_dir"] is None  # local path stripped
    assert info["settings"]["output_dir"] is None  # private path stripped
    assert info["scores"][0]["name"] == "Refusals"

    # Full summary metrics are recorded (filled from the trial's user attrs).
    metrics = info["metrics"]
    assert metrics["initial_refusals"] == 9
    assert metrics["final_refusals"] == 1
    assert metrics["total_prompts"] == 10
    assert metrics["kl_divergence"] == 0.0212
    assert metrics["best_trial"] == 7
    assert metrics["export_strategy"] == "merge"

    # README and SHA256SUMS contents
    sums = (bundle_dir / "SHA256SUMS").read_text(encoding="utf-8")
    assert "model-00001-of-00002.safetensors" in sums
    readme = (bundle_dir / "README.md").read_text(encoding="utf-8")
    assert "Reproduction guide" in readme
    assert "--reproduce" in readme

    # No checkpoint recorded: no journal copy expected.
    assert not list(bundle_dir.glob("*.jsonl"))


def test_settings_restore_from_reproduction(tmp_path):
    settings = Settings(model="example/model", seed=123, abliteration_trials=55,
                        output_dir=tmp_path, cache_dir=tmp_path)
    data = settings.to_dict()
    restored = settings_from_reproduction(data)
    assert isinstance(restored, Settings)
    assert restored.model == "example/model"
    assert restored.seed == 123
    assert restored.abliteration_trials == 55

    # Unknown keys from other versions are ignored.
    restored_unknown = settings_from_reproduction({**data, "future_field": 1})
    assert isinstance(restored_unknown, Settings)

    # All Settings fields survive a bundle round-trip.
    for field in dataclass_fields(Settings):
        assert hasattr(restored, field.name)


def test_restore_keeps_display_flags_and_local_paths(tmp_path):
    """Reproduction restores run-identity settings, but display/diagnostic
    flags and local paths must survive from the command line."""
    from types import SimpleNamespace

    from anlord.pipeline import AbliterationPipeline

    local = Settings(
        model="cli/model",
        print_residual_geometry=True,
        plot_residuals=True,
        output_dir=tmp_path,
    )
    holder = SimpleNamespace(settings=local)
    restored = Settings(
        model="bundle/model",
        seed=999,
        abliteration_trials=77,
        print_residual_geometry=False,
        plot_residuals=False,
        output_dir="C:/original/machine/output",
    )
    AbliterationPipeline._restore_settings_from_reproduction(holder, restored)

    # Run-identity settings are restored from the bundle...
    assert local.model == "bundle/model"
    assert local.seed == 999
    assert local.abliteration_trials == 77
    # ...while display flags and local paths are kept from the command line.
    assert local.print_residual_geometry is True
    assert local.plot_residuals is True
    assert local.output_dir == tmp_path


def test_check_reproduction_environment_same_machine(tmp_path, capsys):
    from anlord.native.reproduce import check_reproduction_environment

    native_cfg = _make_native_config(tmp_path)
    settings = Settings(model=native_cfg.model)

    # Build a bundle for this machine, then check against itself.
    model_dir = tmp_path / "model"
    hashes = _write_weight_files(model_dir)
    bundle_dir = create_reproduction_folder(
        model_dir,
        native_cfg,
        settings,
        checkpoint_path=None,
        trial=_FakeTrial(),
        model_hashes=hashes,
        include_system_information=True,
    )
    info = load_reproduction_information(str(bundle_dir / "reproduce.json"))

    # Same machine, source checkout (distribution not installed): the tool's
    # undeterminable version must not produce a phantom critical mismatch, and
    # the distribution rename must not show up as a second phantom package.
    assert check_reproduction_environment(native_cfg, info) is True
    output = capsys.readouterr().out
    assert "Package Mismatches" not in output
    assert "System Mismatches" not in output

    # Even with ignore_mismatches=False there is nothing to confirm.
    cfg_strict = NativeConfig(**{**native_cfg.__dict__, "ignore_mismatches": False})
    assert check_reproduction_environment(cfg_strict, info) is True


def test_resolve_reproduction_source(tmp_path):
    from anlord.native.reproduce import resolve_reproduction_source

    # Local file path (existing) -> path.
    local = tmp_path / "reproduce.json"
    local.write_text("{}", encoding="utf-8")
    assert resolve_reproduction_source(str(local)) == ("path", str(local))

    # Direct raw URL -> url.
    assert resolve_reproduction_source("https://example.com/reproduce.json") == (
        "url",
        "https://example.com/reproduce.json",
    )

    # Hugging Face repository ID -> repo_id.
    assert resolve_reproduction_source("username/model") == ("repo_id", "username/model")
    assert resolve_reproduction_source("username/model-name_X") == (
        "repo_id",
        "username/model-name_X",
    )

    # Hugging Face model URL (any deep path) -> repo_id.
    assert resolve_reproduction_source("https://huggingface.co/username/model") == (
        "repo_id",
        "username/model",
    )
    assert resolve_reproduction_source(
        "https://huggingface.co/username/model/blob/main/reproduce/reproduce.json"
    ) == ("repo_id", "username/model")

    # A .json reference that is not an existing path stays a path, not a repo.
    assert resolve_reproduction_source("missing/reproduce.json") == (
        "path",
        "missing/reproduce.json",
    )

    # Multi-segment paths are not repo ids.
    assert resolve_reproduction_source("a/b/c") == ("path", "a/b/c")
