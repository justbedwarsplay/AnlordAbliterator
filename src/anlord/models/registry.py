# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Model registry for Anlord Abliterator.
Manages known model configurations and compatibility.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelInfo:
    """Information about a known model."""

    model_id: str
    architecture: str
    parameter_count: int
    recommended_dtype: str = "auto"
    recommended_quantization: str = "none"
    requires_trust_remote_code: bool = False
    notes: str = ""


class ModelRegistry:
    """
    Registry of known models and their configurations.

    This registry helps configure the pipeline for specific models
    and provides compatibility information.
    """

    # Known model configurations
    KNOWN_MODELS = {
        # GPT-OSS models
        "unsloth/gpt-oss-20b-BF16": ModelInfo(
            model_id="unsloth/gpt-oss-20b-BF16",
            architecture="GptOssForCausalLM",
            parameter_count=20_000_000_000,
            recommended_dtype="bfloat16",
            recommended_quantization="none",
            requires_trust_remote_code=True,
            notes="Uses MXFP4 quantization, requires PyTorch 2.6+",
        ),
        "openai/gpt-oss-20b": ModelInfo(
            model_id="openai/gpt-oss-20b",
            architecture="GptOssForCausalLM",
            parameter_count=20_000_000_000,
            recommended_dtype="bfloat16",
            recommended_quantization="none",
            requires_trust_remote_code=True,
            notes="Uses MXFP4 quantization, requires PyTorch 2.6+",
        ),
        # Llama models
        "meta-llama/Llama-3.2-1B": ModelInfo(
            model_id="meta-llama/Llama-3.2-1B",
            architecture="LlamaForCausalLM",
            parameter_count=1_000_000_000,
            recommended_dtype="auto",
            recommended_quantization="none",
        ),
        "meta-llama/Llama-3.2-3B": ModelInfo(
            model_id="meta-llama/Llama-3.2-3B",
            architecture="LlamaForCausalLM",
            parameter_count=3_000_000_000,
            recommended_dtype="auto",
            recommended_quantization="none",
        ),
        # Qwen models
        "Qwen/Qwen2.5-7B": ModelInfo(
            model_id="Qwen/Qwen2.5-7B",
            architecture="Qwen2ForCausalLM",
            parameter_count=7_000_000_000,
            recommended_dtype="auto",
            recommended_quantization="bnb_4bit",
        ),
        "Qwen/Qwen2.5-14B": ModelInfo(
            model_id="Qwen/Qwen2.5-14B",
            architecture="Qwen2ForCausalLM",
            parameter_count=14_000_000_000,
            recommended_dtype="auto",
            recommended_quantization="bnb_4bit",
        ),
        # Gemma models
        "google/gemma-3-12b-it": ModelInfo(
            model_id="google/gemma-3-12b-it",
            architecture="Gemma3ForCausalLM",
            parameter_count=12_000_000_000,
            recommended_dtype="auto",
            recommended_quantization="none",
            notes="Well-supported by Abliteration",
        ),
        # Mistral models
        "mistralai/Mistral-Nemo-Instruct-2407": ModelInfo(
            model_id="mistralai/Mistral-Nemo-Instruct-2407",
            architecture="MistralForCausalLM",
            parameter_count=12_000_000_000,
            recommended_dtype="auto",
            recommended_quantization="bnb_4bit",
        ),
    }

    @classmethod
    def get_info(cls, model_id: str) -> Optional[ModelInfo]:
        """
        Get information about a known model.

        Args:
            model_id: Model ID

        Returns:
            ModelInfo if model is known, None otherwise
        """
        return cls.KNOWN_MODELS.get(model_id)

    @classmethod
    def is_known(cls, model_id: str) -> bool:
        """Check if model is in the registry."""
        return model_id in cls.KNOWN_MODELS

    @classmethod
    def get_all_models(cls) -> list[str]:
        """Get list of all known model IDs."""
        return list(cls.KNOWN_MODELS.keys())
