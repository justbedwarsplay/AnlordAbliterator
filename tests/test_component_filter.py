# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the component include list (e.g. attention-only ablation)."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))
from test_native_parity import TINY_MODEL, _ensure_tiny_model  # noqa: E402

_ensure_tiny_model()


def _make_config(components=None):
    from anlord.native.config import NativeConfig, QuantizationMethod, RowNormalization

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
        abliteration_components=components,
    )


def test_all_components_by_default():
    from anlord.native.model import Model

    model = Model(_make_config())
    assert model.get_abliterable_components() == ["attn.o_proj", "mlp.down_proj"]
    # LoRA wraps both component types.
    lora_names = [
        name for name, _ in model.model.named_modules() if name.endswith("lora_A.default")
    ]
    assert any("down_proj" in name for name in lora_names)
    assert any("o_proj" in name for name in lora_names)


def test_attention_only_filter():
    from anlord.native.model import AbliterationParameters, Model

    model = Model(_make_config(["attn.o_proj"]))
    assert model.get_abliterable_components() == ["attn.o_proj"]
    # LoRA wraps attention only — MLP modules are not even adapted.
    lora_names = [
        name for name, _ in model.model.named_modules() if name.endswith("lora_A.default")
    ]
    assert lora_names
    assert all("down_proj" not in name for name in lora_names)

    # Abliteration works with attention parameters only and only touches
    # attention LoRA weights.
    directions = torch.nn.functional.normalize(
        torch.randn(5, 128), p=2, dim=1
    )
    params = {
        "attn.o_proj": AbliterationParameters(
            max_weight=1.0, max_weight_position=2.0, min_weight=0.3, min_weight_distance=2.0
        )
    }
    model.abliterate(directions, None, params)
    o_proj_set = any(
        "o_proj" in name
        and name.endswith("lora_B.default")
        and module.weight.abs().max().item() > 1e-6
        for name, module in model.model.named_modules()
    )
    assert o_proj_set


def test_prefix_matching():
    from anlord.native.model import Model

    model = Model(_make_config(["attn"]))
    assert model.get_abliterable_components() == ["attn.o_proj"]

    model = Model(_make_config(["mlp"]))
    assert model.get_abliterable_components() == ["mlp.down_proj"]


def test_unknown_component_rejected():
    from anlord.native.model import Model

    with pytest.raises(ValueError, match="match this model's abliterable"):
        Model(_make_config(["does_not_exist"]))


def test_empty_include_list_rejected():
    from anlord.config import Settings

    with pytest.raises(ValueError, match="cannot be empty"):
        Settings(abliteration_components=[])
    # String form is normalized into a list.
    settings = Settings(abliteration_components="attn")
    assert settings.abliteration_components == ["attn"]
