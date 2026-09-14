# SPDX-License-Identifier: AGPL-3.0-or-later
"""Regression tests for lm-evaluation-harness integration."""

from __future__ import annotations

import sys
from types import SimpleNamespace

from anlord.benchmarks.runner import BenchmarkRunner


def test_model_args_do_not_duplicate_simple_evaluate_arguments(tmp_path):
    runner = BenchmarkRunner(
        model_id="example/model",
        output_dir=tmp_path,
        batch_size=4,
        seed=7,
    )

    model_args = runner._get_model_args()

    assert model_args["pretrained"] == "example/model"
    assert "batch_size" not in model_args
    assert "device" not in model_args
    assert "seed" not in model_args


def test_four_bit_is_not_passed_as_hflm_constructor_kwarg(tmp_path):
    runner = BenchmarkRunner(
        model_id="example/model",
        output_dir=tmp_path,
        quantization="bnb_4bit",
        device_map="cuda",
        dtype="bfloat16",
    )

    model_args = runner._get_model_args()
    quant = runner._quantization_config_kwargs(four_bit=True)

    assert "load_in_4bit" not in model_args
    assert "quantization_config" not in model_args
    assert "device_map" not in model_args
    assert quant["load_in_4bit"] is True
    assert quant["llm_int8_enable_fp32_cpu_offload"] is True
    assert quant["bnb_4bit_compute_dtype"] == "bfloat16"


def test_load_in_4bit_constructor_error_stops_remaining_tasks():
    error = "Qwen3_5ForCausalLM.__init__() got an unexpected keyword argument 'load_in_4bit'"

    assert BenchmarkRunner._is_model_initialization_error(error)


def test_duplicate_quantization_config_is_a_model_init_error():
    error = (
        "lm_eval.models.huggingface.HFLM._create_model() got multiple values "
        "for keyword argument 'quantization_config'"
    )

    assert BenchmarkRunner._is_model_initialization_error(error)


def test_metric_extraction_supports_current_lm_eval_schema(tmp_path, monkeypatch):
    captured = {}

    def simple_evaluate(**kwargs):
        captured.update(kwargs)
        return {
            "results": {
                "hellaswag": {
                    "alias": "hellaswag",
                    "acc,none": 0.51,
                    "acc_stderr,none": 0.02,
                    "acc_norm,none": 0.63,
                    "acc_norm_stderr,none": 0.03,
                }
            },
            "n-samples": {"hellaswag": {"original": 10000, "effective": 25}},
        }

    monkeypatch.setitem(sys.modules, "lm_eval", SimpleNamespace(simple_evaluate=simple_evaluate))
    runner = BenchmarkRunner("example/model", tmp_path, batch_size=2, seed=11)
    fake_model, fake_tokenizer = object(), object()
    monkeypatch.setattr(runner, "_ensure_model", lambda: (fake_model, fake_tokenizer))

    result = runner.run_benchmark("hellaswag", limit=25)

    assert result.error is None
    assert result.primary_metric == 0.63
    assert result.primary_metric_stderr == 0.03
    assert result.num_samples == 25
    assert captured["batch_size"] == 2
    assert captured["model_args"]["pretrained"] is fake_model
    assert captured["model_args"]["tokenizer"] is fake_tokenizer
    assert "batch_size" not in captured["model_args"]
    assert "quantization_config" not in captured["model_args"]
    assert captured["torch_random_seed"] == 11


def test_native_library_failure_is_treated_as_model_initialization_error():
    error = r"Could not load this library: C:\site-packages\torchaudio\lib\libtorchaudio.pyd"

    assert BenchmarkRunner._is_model_initialization_error(error)


def test_truthfulqa_alias_uses_registered_mc2_task(tmp_path, monkeypatch):
    captured = {}

    def simple_evaluate(**kwargs):
        captured.update(kwargs)
        return {
            "results": {"truthfulqa_mc2": {"acc,none": 0.75}},
            "n-samples": {"truthfulqa_mc2": {"effective": 10}},
        }

    monkeypatch.setitem(sys.modules, "lm_eval", SimpleNamespace(simple_evaluate=simple_evaluate))
    runner = BenchmarkRunner("example/model", tmp_path)
    monkeypatch.setattr(runner, "_ensure_model", lambda: (object(), object()))
    result = runner.run_benchmark("truthfulqa")

    assert result.error is None
    assert result.task_id == "truthfulqa"
    assert result.primary_metric == 0.75
    assert captured["tasks"] == ["truthfulqa_mc2"]


def test_dump_config_cuda_oom_is_not_a_model_initialization_error():
    error = (
        "CUDA out of memory. Tried to allocate 1.89 GiB. "
        "File lm_eval/api/task.py, in dump_config"
    )

    assert BenchmarkRunner._is_model_initialization_error(error) is False


def test_ensure_model_returns_cached_instance(tmp_path):
    runner = BenchmarkRunner("example/model", tmp_path)
    model, tokenizer = object(), object()
    runner._loaded_model = model
    runner._loaded_tokenizer = tokenizer

    assert runner._ensure_model()[0] is model
    assert runner._ensure_model()[1] is tokenizer
