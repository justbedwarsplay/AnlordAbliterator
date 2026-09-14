# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Model compatibility checker for Heretic.
Checks if models are supported by Heretic's abliteration.
"""

import logging
import pathlib
from typing import Tuple

logger = logging.getLogger(__name__)


# Known supported architectures by Heretic
# Based on Heretic documentation and source code analysis
SUPPORTED_ARCHITECTURES = {
    # Dense Transformer Models
    "LlamaForCausalLM",
    "MistralForCausalLM",
    "MixtralForCausalLM",
    "Qwen2ForCausalLM",
    "Qwen2MoeForCausalLM",
    "Qwen3ForCausalLM",
    "Qwen3MoeForCausalLM",
    "Qwen3_5ForCausalLM",
    "Qwen3_5ForConditionalGeneration",
    "GemmaForCausalLM",
    "Gemma2ForCausalLM",
    "Gemma3ForCausalLM",
    "GPT2LMHeadModel",
    "OPTForCausalLM",
    "FalconForCausalLM",
    "MptForCausalLM",
    "RWForCausalLM",
    "PhiForCausalLM",
    "Phi3ForCausalLM",
    "Starcoder2ForCausalLM",
    # GPT-OSS models (Transformers uses this exact class name)
    "GptOssForCausalLM",
    # Multimodal Models
    "LlavaForConditionalGeneration",
    "LlavaNextForConditionalGeneration",
    "Qwen2VLForConditionalGeneration",
    "Qwen2AudioEncoder",
    # MoE Models
    "MixtralForCausalLM",  # Already listed above
    "Qwen2MoeForCausalLM",  # Already listed above
    "Qwen3MoeForCausalLM",  # Already listed above
    "DeepseekV2ForCausalLM",
    "DeepseekV3ForCausalLM",
    # Hybrid Models
    "Qwen3ForCausalLM",  # Has MoE variants
}


# Model families that are known to work with Heretic
SUPPORTED_MODEL_FAMILIES = {
    # Llama family
    "meta-llama",
    "NousResearch",
    "cognitivecomputations",
    "huginn",
    # Mistral family
    "mistralai",
    "mistral",
    "mmnga",
    # Qwen family
    "Qwen",
    "Qwen2",
    "Qwen3",
    "huihui-ai",
    "ornith-ai",
    # Gemma family
    "google",
    "google/gemma",
    # GPT-OSS
    "openai",
    "unsloth",
    # Others
    "EleutherAI",
    "tiiuae",
    "bigscience",
    "bigcode",
    "microsoft",
    "openchat",
    "AlignmentLab",
    "mlabonne",
    "p-e-w",
}


# Known unsupported architectures
UNSUPPORTED_ARCHITECTURES = {
    # State Space Models
    "MambaForCausalLM",
    "Mamba2ForCausalLM",
    "JambaForCausalLM",
    # Encoder-only models
    "BertModel",
    "RobertaModel",
    "DebertaV2Model",
    # Encoder-decoder models
    "T5ForConditionalGeneration",
    "MT5ForConditionalGeneration",
    "BartForConditionalGeneration",
    "BigBirdPegasusForConditionalGeneration",
    # Vision models without language
    "CLIPVisionModel",
    "ViTForImageClassification",
}


def check_model_compatibility(
    architecture: str | None,
    model_type: str | None,
    model_id: str | None = None,
) -> Tuple[bool, str]:
    """
    Check if a model architecture is compatible with Heretic.

    Args:
        architecture: The architecture class name (e.g., "LlamaForCausalLM")
        model_type: The model type string (e.g., "llama")
        model_id: Optional model ID for additional checks

    Returns:
        Tuple of (is_compatible, message)
    """
    if not architecture and not model_type:
        return False, "Could not determine model architecture"

    # Check explicitly unsupported
    if architecture in UNSUPPORTED_ARCHITECTURES:
        return False, f"Architecture {architecture} is not supported by Heretic"

    # Check explicitly supported
    if architecture in SUPPORTED_ARCHITECTURES:
        logger.info(f"Architecture {architecture} is supported")
        return True, f"Architecture {architecture} is supported"

    # Check model families
    if model_id:
        model_id_lower = model_id.lower()
        for family in SUPPORTED_MODEL_FAMILIES:
            if family.lower() in model_id_lower:
                logger.info(f"Model family {family} is supported")
                return True, f"Model family {family} is supported"

    # For unknown architectures, log warning but allow attempt
    warning_msg = (
        f"Architecture {architecture} (type: {model_type}) is not explicitly "
        "verified to work with Heretic. Attempting anyway..."
    )
    logger.warning(warning_msg)
    return True, warning_msg


def check_model_requires_special_handling(
    architecture: str | None,
    model_id: str | None,
) -> Tuple[bool, str]:
    """
    Check if a model requires special handling.

    Args:
        architecture: The architecture class name
        model_id: The model ID

    Returns:
        Tuple of (requires_special_handling, reason)
    """
    if not model_id:
        return False, ""

    model_id_lower = model_id.lower()

    # GPT-OSS models need special handling
    if "gpt-oss" in model_id_lower or "gptoss" in model_id_lower:
        return True, "GPT-OSS models use MXFP4 quantization and require special handling"

    # 4-bit quantized models
    if "4bit" in model_id_lower or "bnb_4bit" in model_id_lower:
        return True, "4-bit quantized models require bitsandbytes"

    # MoE models
    if any(x in model_id_lower for x in ["moe", "mixture"]):
        return True, "Mixture of Experts models may require additional memory"

    return False, ""


def estimate_vram_requirements(
    model_id: str,
    architecture: str | None = None,
    dtype: str = "bfloat16",
) -> float:
    """
    Estimate VRAM requirements for a model.

    Args:
        model_id: Model ID or path
        architecture: Optional architecture class name
        dtype: Data type for computation

    Returns:
        Estimated VRAM in GB
    """
    # Try to get parameter count from config
    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

        # Get hidden size and other parameters
        hidden_size = getattr(config, "hidden_size", 0)
        num_layers = getattr(config, "num_hidden_layers", 0)
        vocab_size = getattr(config, "vocab_size", 0)
        intermediate_size = getattr(config, "intermediate_size", 0)

        # Estimate parameters (very rough)
        # For decoder-only models:
        # embedding: vocab_size * hidden_size * 2 (embed + output)
        # attention: num_layers * hidden_size * hidden_size * 4
        # ffn: num_layers * hidden_size * intermediate_size * 2

        embedding_params = vocab_size * hidden_size * 2
        attention_params = num_layers * hidden_size * hidden_size * 4
        ffn_params = num_layers * hidden_size * intermediate_size * 2

        total_params = embedding_params + attention_params + ffn_params

        # Convert to bytes based on dtype
        bytes_per_param = {
            "float32": 4,
            "float16": 2,
            "bfloat16": 2,
            "auto": 2,  # Usually bfloat16
        }.get(dtype, 2)

        vram_gb = (total_params * bytes_per_param) / (1024**3)

        # Add 20% overhead for activations, KV cache, etc.
        vram_gb *= 1.2

        return vram_gb

    except Exception as e:
        logger.warning(f"Could not estimate VRAM: {e}")
        return 0.0

# Task 2: Gated DeltaNet fast path check (Qwen3.5 etc)
def _get_layer_types_from_config(config) -> list | None:
    """Extract layer_types whether top-level or inside text_config."""
    if config is None:
        return None
    # text_config.layer_types is used by Qwen3.5
    text_cfg = getattr(config, "text_config", None)
    if text_cfg is not None:
        lt = getattr(text_cfg, "layer_types", None)
        if lt is not None:
            return list(lt)
    lt = getattr(config, "layer_types", None)
    if lt is not None:
        return list(lt)
    return None

def check_delta_fast_path(
    model_id: str,
    model_commit: str | None = None,
    cache_dir: pathlib.Path | str | None = None,
) -> dict:
    """Check whether model is DeltaNet hybrid and fast kernels are available.

    Returns dict with keys:
      is_delta_model, layer_types, causal_available, fla_available,
      fast_path_available, triton_available
    """
    import pathlib as _pl
    result: dict = {
        "is_delta_model": False,
        "layer_types": [],
        "causal_available": False,
        "fla_available": False,
        "fast_path_available": True,
        "triton_available": False,
    }
    # availability checks
    try:
        from transformers.utils import import_utils as iu
        try:
            result["causal_available"] = bool(iu.is_causal_conv1d_available())
        except Exception:
            result["causal_available"] = False
        try:
            result["fla_available"] = bool(iu.is_flash_linear_attention_available())
        except Exception:
            result["fla_available"] = False
        try:
            # triton is required for fla on some platforms
            import importlib.util as _iu
            result["triton_available"] = _iu.find_spec("triton") is not None
        except Exception:
            pass
    except Exception as e:
        logger.debug(f"check_delta_fast_path availability check failed: {e}")
    # load config
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(
            model_id,
            revision=model_commit,
            cache_dir=(str(_pl.Path(cache_dir) / "hub") if cache_dir else None),
            trust_remote_code=True,
        )
        layer_types = _get_layer_types_from_config(cfg)
        if layer_types:
            result["layer_types"] = layer_types
            if "linear_attention" in layer_types:
                result["is_delta_model"] = True
                result["fast_path_available"] = bool(result["causal_available"] and result["fla_available"])
            else:
                result["is_delta_model"] = False
                result["fast_path_available"] = True
        else:
            result["is_delta_model"] = False
            result["fast_path_available"] = True
    except Exception as e:
        logger.debug(f"check_delta_fast_path config load failed: {e}")
        # if we cannot load config, assume not delta to avoid false warning
        result["fast_path_available"] = True
    return result
