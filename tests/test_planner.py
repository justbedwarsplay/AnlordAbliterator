# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for hardware-aware model load planning."""

from anlord.hardware.planner import (
    estimate_weight_gb_from_name,
    explain_model_load_failure,
    heretic_device_map_for,
    heretic_dtypes_for,
    looks_like_access_violation,
    plan_model_load,
)


def test_gpt_oss_name_estimates_about_40gb_bf16():
    assert estimate_weight_gb_from_name("unsloth/gpt-oss-20b-BF16") == 40.0


def test_auto_quantization_enables_4bit_for_20b_on_8gb_laptop():
    plan = plan_model_load(
        model_id="unsloth/gpt-oss-20b-BF16",
        device="cuda",
        dtype="auto",
        quantization="auto",
        weight_gb=41.8,
        vram_gb=8.0,
        ram_gb=16.0,
        available_ram_gb=8.0,
        commit_limit_gb=24.0,
        platform_name="posix",
    )

    assert plan.quantization == "bnb_4bit"
    assert plan.device_map == "auto"
    assert plan.max_memory is not None
    assert "0" in plan.max_memory
    assert "cpu" in plan.max_memory
    assert plan.heretic_batch_size == 1
    assert plan.heretic_dtypes == ["auto", "bfloat16", "float16"]


def test_windows_auto_avoids_mixed_4bit_offload_for_20b():
    plan = plan_model_load(
        model_id="unsloth/gpt-oss-20b-BF16",
        device="cuda",
        dtype="auto",
        quantization="auto",
        weight_gb=41.8,
        vram_gb=8.0,
        ram_gb=16.0,
        available_ram_gb=8.0,
        commit_limit_gb=112.0,
        platform_name="nt",
    )

    assert plan.quantization == "none"
    assert plan.device_map == "auto"
    assert plan.max_memory is not None
    assert "cpu" in plan.max_memory
    assert any("0xC0000005" in reason for reason in plan.reasons)


def test_nine_b_four_bit_stays_on_gpu_without_cpu_split():
    plan = plan_model_load(
        model_id="ornith-ai/Ornith-1.5-9B",
        device="cuda",
        dtype="bfloat16",
        quantization="bnb_4bit",
        weight_gb=18.0,
        vram_gb=8.0,
        ram_gb=16.0,
        available_ram_gb=4.0,
        commit_limit_gb=23.0,
        platform_name="nt",
    )

    assert plan.quantization == "bnb_4bit"
    assert plan.device_map == "cuda"
    assert plan.max_memory is None


def test_explicit_none_is_respected_and_offloads_to_cpu():
    plan = plan_model_load(
        model_id="unsloth/gpt-oss-20b-BF16",
        device="cuda",
        dtype="bfloat16",
        quantization="none",
        weight_gb=41.8,
        vram_gb=8.0,
        ram_gb=16.0,
        available_ram_gb=8.0,
        commit_limit_gb=112.0,
    )

    assert plan.quantization == "none"
    assert plan.device_map == "auto"
    assert plan.max_memory is not None
    assert "cpu" in plan.max_memory
    assert any("quantization=none keeps full precision" in warning for warning in plan.warnings)


def test_small_model_stays_full_precision():
    plan = plan_model_load(
        model_id="Qwen/Qwen2.5-0.5B",
        device="cuda",
        dtype="auto",
        quantization="auto",
        weight_gb=1.0,
        vram_gb=8.0,
        ram_gb=16.0,
        available_ram_gb=10.0,
        commit_limit_gb=24.0,
    )

    assert plan.quantization == "none"


def test_cuda_device_uses_accelerate_auto_map():
    assert heretic_device_map_for("cuda") == "auto"
    assert heretic_device_map_for("cpu") == "cpu"


def test_auto_dtype_does_not_collapse_to_a_single_attempt():
    assert heretic_dtypes_for("auto") == ["auto", "bfloat16", "float16"]


def test_pagefile_error_is_explained_without_rich_traceback():
    output = """
* Trying dtype bfloat16...
* Failed (Файл подкачки слишком мал для завершения операции. (os error 1455))
┌───────────────────── Traceback
│ model.py:169
Exception: Failed to load model with all configured dtypes.
"""

    message = explain_model_load_failure(output, 1)

    assert "os error 1455" in message
    assert "paging file" in message.lower() or "Virtual memory" in message
    assert "Traceback" not in message
    assert "model.py:169" not in message


def test_bitsandbytes_cpu_dispatch_error_is_explained():
    output = (
        "* Failed (Some modules are dispatched on the CPU or the disk. "
        "Make sure you have enough GPU RAM to fit the quantized model.)"
    )
    message = explain_model_load_failure(output, 1)
    assert "llm_int8_enable_fp32_cpu_offload" in message


def test_meta_tensor_error_is_explained():
    output = "NotImplementedError: Cannot copy out of meta tensor; no data!"
    message = explain_model_load_failure(output, 1)
    assert "meta-tensor" in message
    assert "device_map=cuda" in message


def test_windows_access_violation_exit_code_is_explained():
    assert looks_like_access_violation(3221225477)
    message = explain_model_load_failure("", 3221225477)
    assert "0xC0000005" in message
    assert "bitsandbytes" in message
    assert "without quantization" not in message
    assert "--no-resume" not in message
    assert "CausalLM" in message
