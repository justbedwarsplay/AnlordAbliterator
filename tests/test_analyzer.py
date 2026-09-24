# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the residual analyzer (geometry printing + PaCMAP plots)."""

import json
from pathlib import Path

import pytest
import torch

pytest.importorskip("torch")

from anlord.native.analyzer import ResidualAnalyzer  # noqa: E402
from anlord.native.config import NativeConfig, QuantizationMethod  # noqa: E402


class _FakeLayers(list):
    pass


class _FakeModel:
    """Just enough of the Model interface for the analyzer: layer count."""

    def __init__(self, n_layers: int):
        self._layers = _FakeLayers([object() for _ in range(n_layers)])

    def get_layers(self):
        return self._layers


def _make_config(tmp_path: Path) -> NativeConfig:
    return NativeConfig(
        model="tiny/test-model",
        dtypes=["float32"],
        quantization=QuantizationMethod.NONE,
        device_map="cpu",
        batch_size=4,
        offload_outputs_to_cpu=False,
        seed=42,
        residual_plot_path=str(tmp_path / "plots"),
    )


@pytest.fixture(scope="module")
def residuals():
    torch.manual_seed(42)
    n_layers_plus_one = 5  # embeddings + 4 layers
    hidden = 64
    # Two well-separated clusters so silhouettes are meaningful.
    good = torch.randn(32, n_layers_plus_one, hidden) * 0.1
    bad = torch.randn(32, n_layers_plus_one, hidden) * 0.1 + 3.0
    return good, bad


def _analyzer(tmp_path, residuals):
    cfg = _make_config(tmp_path)
    model = _FakeModel(n_layers=4)
    return ResidualAnalyzer(cfg, model, residuals[0], residuals[1]), cfg


def test_print_residual_geometry(tmp_path, residuals, capsys):
    pytest.importorskip("geom_median")
    pytest.importorskip("sklearn")

    analyzer, cfg = _analyzer(tmp_path, residuals)
    analyzer.print_residual_geometry()
    output = capsys.readouterr().out
    assert "Residual Geometry" in output
    assert "geometric median" in output
    # Table has one row per transformer layer (embeddings row is skipped).
    assert "1" in output and "4" in output
    assert "Silh" in output
    # The full table is saved next to the plots: text + machine-readable JSON.
    base_path = Path(cfg.residual_plot_path) / "tiny_test-model"
    text_file = base_path / "residual_geometry.txt"
    json_file = base_path / "residual_geometry.json"
    assert text_file.is_file() and json_file.is_file()
    text = text_file.read_text(encoding="utf-8")
    assert "S(g*,r*)" in text  # columns not truncated in the file rendering
    geometry = json.loads(json_file.read_text(encoding="utf-8"))
    assert geometry["model"] == "tiny/test-model"
    assert [entry["layer"] for entry in geometry["layers"]] == [1, 2, 3, 4]
    for entry in geometry["layers"]:
        assert 0.0 <= entry["Silh"] <= 1.0
        assert -1.0 <= entry["S(g,b)"] <= 1.0


def test_plot_residuals(tmp_path, residuals):
    pytest.importorskip("pacmap")
    pytest.importorskip("imageio")
    pytest.importorskip("matplotlib")
    pytest.importorskip("geom_median")

    analyzer, cfg = _analyzer(tmp_path, residuals)
    analyzer.plot_residuals()

    base_path = Path(cfg.residual_plot_path) / "tiny_test-model"
    assert base_path.is_dir()
    # One PNG per layer.
    layer_frames = sorted(base_path.glob("layer_*.png"))
    assert len(layer_frames) == 4
    assert (base_path / "animation.gif").is_file()


def test_research_dependencies_message(tmp_path, residuals, monkeypatch, capsys):
    """Without the research extras, a clear installation hint is printed."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("geom_median") or name.startswith("sklearn"):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    analyzer, _ = _analyzer(tmp_path, residuals)
    analyzer.print_residual_geometry()
    output = capsys.readouterr().out
    assert "Research dependencies not found" in output
    assert "anlord[research]" in output
