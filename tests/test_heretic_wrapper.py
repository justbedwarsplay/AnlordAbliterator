# SPDX-License-Identifier: AGPL-3.0-or-later
"""Regression tests for the Heretic subprocess wrapper."""

from pathlib import Path

from types import SimpleNamespace

from anlord.heretic.bridge import (
    _PromptController,
    enable_generation_progress,
    enable_low_ram_lora_merge,
    enable_quantized_cpu_offload,
    force_text_only_causal_lm,
)
from anlord.heretic.wrapper import HereticWrapper, configure_cuda_allocator_env


class _Trial:
    def __init__(self):
        self.user_attrs = {
            "index": 4,
            "base_refusals": 90,
            "refusals": 3,
            "n_bad_prompts": 100,
            "kl_divergence": 0.12,
        }


def test_output_parser_handles_actual_heretic_format():
    wrapper = object.__new__(HereticWrapper)
    output = """
* Initial refusals: 90/100
  * KL divergence: 0.1200
  * Refusals: 3/100
"""

    assert wrapper._parse_heretic_output(output) == {
        "initial_refusals": 90,
        "final_refusals": 3,
        "total_prompts": 100,
        "kl_divergence": 0.12,
    }


def test_model_directory_must_contain_config_and_weights(tmp_path):
    assert not HereticWrapper.is_model_directory(tmp_path)
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    assert not HereticWrapper.is_model_directory(tmp_path)
    (tmp_path / "model.safetensors").touch()
    assert HereticWrapper.is_model_directory(tmp_path)


def test_four_bit_retry_first_drops_cpu_split(tmp_path, monkeypatch):
    monkeypatch.setattr(HereticWrapper, "_require_heretic", staticmethod(lambda: None))
    wrapper = HereticWrapper(
        model_id="example/model",
        output_dir=tmp_path,
        quantization="bnb_4bit",
        max_memory={"0": "6GB", "cpu": "7GB"},
    )

    assert wrapper._should_retry_without_mixed_offload(
        RuntimeError("Tensor.item() cannot be called on meta tensors")
    )
    wrapper._keep_four_bit_on_gpu_only()
    assert wrapper.quantization == "bnb_4bit"
    assert wrapper.max_memory is None
    assert wrapper.device_map == "cuda"


def test_access_violation_fallback_disables_quantization(tmp_path, monkeypatch):
    monkeypatch.setattr(HereticWrapper, "_require_heretic", staticmethod(lambda: None))
    wrapper = HereticWrapper(
        model_id="example/model",
        output_dir=tmp_path,
        quantization="bnb_4bit",
        max_memory={"0": "7GB", "cpu": "10GB"},
    )
    wrapper._switch_to_full_precision_offload()

    assert wrapper.quantization == "none"
    assert wrapper.device_map == "auto"
    assert wrapper.max_memory["cpu"] == "48GB"


def test_gpu_only_four_bit_does_not_fall_back_to_bf16(tmp_path, monkeypatch):
    monkeypatch.setattr(HereticWrapper, "_require_heretic", staticmethod(lambda: None))
    wrapper = HereticWrapper(
        model_id="ornith-ai/Ornith-1.5-9B",
        output_dir=tmp_path,
        quantization="bnb_4bit",
        device_map="cuda",
        max_memory=None,
    )
    wrapper._retry_pause_seconds = 0
    crash = RuntimeError("Heretic crashed with Windows access violation 0xC0000005")

    assert wrapper._started_gpu_only_four_bit is True
    assert wrapper._apply_load_retry(crash) is True
    assert wrapper.quantization == "bnb_4bit"
    assert wrapper.device_map == "cuda"
    assert wrapper.max_memory is None
    assert wrapper._apply_load_retry(crash) is False
    assert wrapper.quantization == "bnb_4bit"


def test_windows_environment_drops_expandable_segments():
    environment = {
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,max_split_size_mb:128"
    }
    configure_cuda_allocator_env(environment, is_windows=True)
    assert environment.get("PYTORCH_CUDA_ALLOC_CONF") == "max_split_size_mb:128"

    empty = {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
    configure_cuda_allocator_env(empty, is_windows=True)
    assert "PYTORCH_CUDA_ALLOC_CONF" not in empty

    linux = {}
    configure_cuda_allocator_env(linux, is_windows=False)
    assert linux["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"


def test_environment_uses_auto_device_map_and_quantization(tmp_path, monkeypatch):
    monkeypatch.setattr(HereticWrapper, "_require_heretic", staticmethod(lambda: None))
    wrapper = HereticWrapper(
        model_id="unsloth/gpt-oss-20b-BF16",
        output_dir=tmp_path,
        device="cuda",
        dtype="auto",
        quantization="bnb_4bit",
        max_memory={"0": "7GB", "cpu": "10GB"},
        heretic_batch_size=1,
    )
    environment = wrapper._environment()

    assert environment["HERETIC_DEVICE_MAP"] == "auto"
    assert environment["HERETIC_QUANTIZATION"] == "bnb_4bit"
    assert environment["HERETIC_DTYPES"] == '["auto", "bfloat16", "float16"]'
    assert environment["HERETIC_MAX_MEMORY"] == '{"0": "7GB", "cpu": "10GB"}'
    assert environment["HERETIC_BATCH_SIZE"] == "1"
    assert environment["HERETIC_OFFLOAD_OUTPUTS_TO_CPU"] == "true"
    assert "HERETIC_MAX_RESPONSE_LENGTH" not in environment
    assert "ANLORD_PREFIX_CHECK" not in environment


def test_bridge_enables_bitsandbytes_cpu_offload():
    class FakeConfig:
        def __init__(self):
            self.llm_int8_enable_fp32_cpu_offload = False

    class FakeModel:
        def _get_quantization_config(self, dtype):
            return FakeConfig()

    module = SimpleNamespace(Model=FakeModel)
    enable_quantized_cpu_offload(module)
    config = module.Model()._get_quantization_config("bfloat16")

    assert config.llm_int8_enable_fp32_cpu_offload is True


def test_gpu_only_quantization_does_not_enable_cpu_offload(monkeypatch):
    monkeypatch.setenv("HERETIC_DEVICE_MAP", "cuda")
    monkeypatch.delenv("HERETIC_MAX_MEMORY", raising=False)

    class FakeConfig:
        def __init__(self):
            self.llm_int8_enable_fp32_cpu_offload = True
            self.llm_int8_skip_modules = None
            self.bnb_4bit_use_double_quant = True

    class FakeModel:
        def _get_quantization_config(self, dtype):
            return FakeConfig()

    module = SimpleNamespace(Model=FakeModel)
    enable_quantized_cpu_offload(module)
    config = module.Model()._get_quantization_config("bfloat16")

    assert config.llm_int8_enable_fp32_cpu_offload is False
    assert "visual" in config.llm_int8_skip_modules


def test_bridge_forces_text_only_causal_lm(monkeypatch):
    class FakeCausal:
        pass

    fake_transformers = SimpleNamespace(AutoModelForCausalLM=FakeCausal)
    monkeypatch.setitem(__import__("sys").modules, "transformers", fake_transformers)
    module = SimpleNamespace(get_model_class=lambda model: "vl")
    force_text_only_causal_lm(module)

    assert module.get_model_class("ornith-ai/Ornith-1.5-9B") is FakeCausal


def test_low_ram_merge_releases_gpu_model_before_reload(monkeypatch):
    class FakeAuto:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            del args
            return {"low_cpu_mem_usage": kwargs.get("low_cpu_mem_usage")}

    class FakeModel:
        def __init__(self):
            self.model = object()

        def get_merged_model(self):
            import transformers

            result = transformers.AutoModelForCausalLM.from_pretrained("model")
            return result, self.model

    fake_transformers = SimpleNamespace(AutoModelForCausalLM=FakeAuto)
    monkeypatch.setitem(__import__("sys").modules, "transformers", fake_transformers)
    module = SimpleNamespace(Model=FakeModel)
    enable_low_ram_lora_merge(module)
    holder = module.Model()
    loaded, remaining = holder.get_merged_model()

    assert loaded["low_cpu_mem_usage"] is True
    assert remaining is None


def test_bridge_selects_trial_saves_once_and_exposes_metrics(tmp_path):
    controller = _PromptController(Path(tmp_path) / "model")
    trial = _Trial()

    assert controller.select("Which trial do you want to use?", [trial, ""]) is trial
    assert (
        controller.select(
            "What do you want to do with the decensored model?",
            ["Save the model to a local folder", ""],
        )
        == "Save the model to a local folder"
    )
    assert controller.path() == str(Path(tmp_path) / "model")
    assert controller.select("Which trial do you want to use?", [trial, ""]) == ""
    assert controller.metrics()["final_refusals"] == 3
    assert controller.metrics()["kl_divergence"] == 0.12


def test_generation_progress_keeps_all_prefix_prompts():
    class FakeModel:
        def get_responses(self, prompts, *args, **kwargs):
            return [f"r{index}" for index in range(len(list(prompts)))]

        def get_responses_batched(self, prompts, *args, **kwargs):
            return self.get_responses(prompts, *args, **kwargs)

        def get_residuals(self, prompts, *args, **kwargs):
            return list(prompts)

        def get_residuals_batched(self, prompts, *args, **kwargs):
            return self.get_residuals(prompts, *args, **kwargs)

    module = SimpleNamespace(Model=FakeModel)
    enable_generation_progress(module)
    prompts = list(range(200))
    model = module.Model()

    responses = model.get_responses_batched(prompts)
    residuals = model.get_residuals_batched(prompts)

    assert len(responses) == 200
    assert len(residuals) == 200
