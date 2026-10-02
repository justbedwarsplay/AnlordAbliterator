# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for 1.6.0: input-side ablation, gaussian decay, multi-token KL."""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from test_native_parity import TINY_MODEL, _ensure_tiny_model  # noqa: E402

_ensure_tiny_model()

from anlord.native.model import (  # noqa: E402
    ablation_weight,
    read_side_delta,
    write_side_delta,
)


# ---------------------------------------------------------------------------
# Delta math
# ---------------------------------------------------------------------------


def test_read_side_delta_blinds_direction():
    """At weight 1 the read-side edit makes the module blind to the direction."""
    torch.manual_seed(0)
    W = torch.randn(8, 16)
    v = F.normalize(torch.randn(16), dim=0)
    W2 = W + read_side_delta(W, v, 1.0)
    assert torch.allclose(W2 @ v, torch.zeros(8), atol=1e-5)


def test_read_side_delta_preserves_other_directions_weakly():
    """The read-side edit scales the direction component by (1 - weight)."""
    torch.manual_seed(0)
    W = torch.randn(8, 16)
    v = F.normalize(torch.randn(16), dim=0)
    W2 = W + read_side_delta(W, v, 0.5)
    assert torch.allclose(W2 @ v, 0.5 * (W @ v), atol=1e-5)


def test_write_side_delta_removes_output_component():
    torch.manual_seed(1)
    W = torch.randn(8, 16)
    v = F.normalize(torch.randn(8), dim=0)
    x = torch.randn(16)
    W2 = W + write_side_delta(W, v, 1.0)
    assert abs(float(v @ (W2 @ x))) < 1e-5


# ---------------------------------------------------------------------------
# Decay kernels
# ---------------------------------------------------------------------------


def test_ablation_weight_linear():
    assert ablation_weight(1.5, 0.5, 0.0, 10.0) == pytest.approx(1.5)
    assert ablation_weight(1.5, 0.5, 10.0, 10.0) == pytest.approx(0.5)
    assert ablation_weight(1.5, 0.5, 5.0, 10.0) == pytest.approx(1.0)


def test_ablation_weight_gaussian():
    # at distance 0 the weight is max; at sigma it is ~60.7% of the range above min
    assert ablation_weight(1.5, 0.5, 0.0, 10.0, "gaussian") == pytest.approx(1.5)
    assert ablation_weight(1.5, 0.5, 10.0, 10.0, "gaussian") == pytest.approx(
        0.5 + 1.0 * 0.6065306597126334, rel=1e-4
    )
    weights = [ablation_weight(1.5, 0.5, d, 10.0, "gaussian") for d in range(11)]
    assert all(a >= b for a, b in zip(weights, weights[1:]))


# ---------------------------------------------------------------------------
# Input-side components on the tiny model
# ---------------------------------------------------------------------------


def _tiny_config(tmp_path, input_side):
    from anlord.native.config import (
        NativeConfig,
        QuantizationMethod,
        RowNormalization,
    )

    return NativeConfig(
        model=str(TINY_MODEL.resolve()),
        dtypes=["float32"],
        quantization=QuantizationMethod.NONE,
        device_map="cpu",
        batch_size=2,
        row_normalization=RowNormalization.NONE,
        offload_outputs_to_cpu=False,
        seed=42,
        response_prefix="",
        abliteration_input_side=input_side,
        study_checkpoint_dir=str(tmp_path / "checkpoints"),
    )


def test_input_side_components_present(tmp_path):
    from anlord.native.model import Model

    model = Model(_tiny_config(tmp_path, input_side=True))
    components = model.get_abliterable_components()
    assert "attn.qkv_in" in components
    assert "mlp.gateup_in" in components
    assert "attn.o_proj" in components
    assert "mlp.down_proj" in components


def test_input_side_components_absent_by_default(tmp_path):
    from anlord.native.model import Model

    model = Model(_tiny_config(tmp_path, input_side=False))
    components = model.get_abliterable_components()
    assert "attn.qkv_in" not in components
    assert "mlp.gateup_in" not in components


# ---------------------------------------------------------------------------
# Multi-token KL
# ---------------------------------------------------------------------------


def test_kl_scorer_token_count(tmp_path):
    from anlord.native.config import DatasetSpecification, NativeConfig
    from anlord.native.model import Model

    good_file = tmp_path / "good.txt"
    good_file.write_text(
        "\n".join(f"Good prompt {i}" for i in range(4)) + "\n", encoding="utf-8"
    )
    cfg = _tiny_config(tmp_path, input_side=False)
    cfg.scorer_settings = {
        "KLDivergence": {
            "prompts": {"dataset": str(good_file), "split": "[:4]"},
            "token_count": 3,
        }
    }
    model = Model(cfg)
    from anlord.native.evaluator import Evaluator

    evaluator = Evaluator(cfg, model)
    kl_entry = next(e for e in evaluator._scorer_entries if e.name == "KL divergence")
    baseline = kl_entry.scorer._baseline_logprobs
    assert baseline.shape[0] == 3  # three token positions
    scores = evaluator.get_scores()
    kl_score = dict(scores)["KL divergence"]
    assert kl_score.value >= 0.0


def test_kl_scorer_default_is_single_token(tmp_path):
    from anlord.native.config import DatasetSpecification, NativeConfig
    from anlord.native.model import Model

    good_file = tmp_path / "good.txt"
    good_file.write_text(
        "\n".join(f"Good prompt {i}" for i in range(4)) + "\n", encoding="utf-8"
    )
    cfg = _tiny_config(tmp_path, input_side=False)
    cfg.good_evaluation_prompts = DatasetSpecification(
        dataset=str(good_file), split="[:4]"
    )
    cfg.scorer_settings = {
        "KLDivergence": {"prompts": {"dataset": str(good_file), "split": "[:4]"}}
    }
    model = Model(cfg)
    from anlord.native.evaluator import Evaluator

    evaluator = Evaluator(cfg, model)
    kl_entry = next(e for e in evaluator._scorer_entries if e.name == "KL divergence")
    assert kl_entry.scorer._baseline_logprobs.shape[0] == 1


# ---------------------------------------------------------------------------
# Settings / CLI wiring
# ---------------------------------------------------------------------------


def test_settings_validation_decay_kernel():
    from anlord.config import Settings

    settings = Settings()
    assert settings.abliteration_input_side is False
    assert settings.abliteration_decay_kernel == "linear"
    with pytest.raises(ValueError, match="decay_kernel"):
        Settings(abliteration_decay_kernel="cubic")


def test_native_config_mapping_160():
    from anlord.config import Settings
    from anlord.native.config import NativeConfig

    settings = Settings(
        abliteration_input_side=True,
        abliteration_decay_kernel="gaussian",
    )
    cfg = NativeConfig.from_anlord_settings(settings)
    assert cfg.abliteration_input_side is True
    assert cfg.decay_kernel == "gaussian"
    assert NativeConfig.from_anlord_settings(Settings()).abliteration_input_side is False


def test_cli_flags_160():
    from anlord.cli.main import create_parser

    parser = create_parser()
    args = parser.parse_args(
        ["--model", "x/model", "--input-side-ablation", "--decay-kernel", "gaussian"]
    )
    assert args.input_side_ablation is True
    assert args.decay_kernel == "gaussian"
    args = parser.parse_args(["--model", "x/model", "--no-input-side-ablation"])
    assert args.input_side_ablation is False


# ---------------------------------------------------------------------------
# Direct steering mode
# ---------------------------------------------------------------------------


def _direct_config(tmp_path):
    from anlord.native.config import (
        NativeConfig,
        QuantizationMethod,
        RowNormalization,
    )

    return NativeConfig(
        model=str(TINY_MODEL.resolve()),
        dtypes=["float32"],
        quantization=QuantizationMethod.NONE,
        device_map="cpu",
        batch_size=2,
        row_normalization=RowNormalization.NONE,
        offload_outputs_to_cpu=False,
        seed=42,
        response_prefix="",
        steering_mode="direct",
        abliteration_input_side=True,
        study_checkpoint_dir=str(tmp_path / "checkpoints"),
    )


def test_direct_mode_edit_and_reload_restore(tmp_path):
    """Direct steering edits base weights in place; reset reloads pristine
    weights from disk - exact state every trial."""
    from anlord.native.model import AbliterationParameters, Model
    from anlord.native.prompts import Prompt

    model = Model(_direct_config(tmp_path))
    original = {n: p.detach().clone() for n, p in model.model.named_parameters()}
    components = model.get_abliterable_components()
    prompts = [Prompt(system="s", user=f"p{i}") for i in range(2)]
    directions = model.get_residuals_batched(prompts).mean(dim=0)
    params = {
        c: AbliterationParameters(
            max_weight=1.0, max_weight_position=2.0, min_weight=0.5, min_weight_distance=4.0
        )
        for c in components
    }
    model.abliterate(directions, 2.0, params)
    changed = sum(
        1
        for n, p in model.model.named_parameters()
        if not torch.equal(p.detach(), original[n])
    )
    assert changed > 0
    model.reset_model()
    exact = sum(
        1
        for n, p in model.model.named_parameters()
        if torch.equal(p.detach(), original[n])
    )
    assert exact == len(original)


def test_direct_mode_requires_quant_none():
    from anlord.native.config import NativeConfig, QuantizationMethod

    cfg = _direct_config(tmp_path=None) if False else None
    # build config then flip quantization
    from test_ablation_160 import _direct_config as _dc  # noqa: F401

    cfg = None
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        cfg = _dc(Path(td))
        cfg.quantization = QuantizationMethod.BNB_4BIT
        from anlord.native.model import Model

        with pytest.raises(Exception, match="Direct steering requires quantization none"):
            Model(cfg)
