# SPDX-License-Identifier: AGPL-3.0-or-later
"""Runtime compatibility helpers for the ML dependency stack."""

from __future__ import annotations

import importlib
import logging
import os
import sys
import warnings
from pathlib import Path

logger = logging.getLogger(__name__)

_OPTIONAL_DEPENDENCIES_PREPARED = False


def configure_huggingface_environment(cache_dir: Path | str) -> Path:
    """Point all Hugging Face libraries at the configured cache directory.

    Some Hugging Face constants are evaluated during module import, so updating
    only ``model_args.cache_dir`` is insufficient for tokenizers and datasets.
    This function updates both the environment and already-imported modules.
    """
    cache_path = Path(cache_dir).resolve()
    hub_cache = cache_path / "hub"
    datasets_cache = cache_path / "datasets"
    cache_path.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(cache_path)
    os.environ["HF_HUB_CACHE"] = str(hub_cache)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub_cache)
    os.environ["HF_DATASETS_CACHE"] = str(datasets_cache)
    if os.name == "nt":
        # Windows works without symlinks; avoid repeating a warning for every file.
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    hub_constants = sys.modules.get("huggingface_hub.constants")
    if hub_constants is not None:
        for name in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
            if hasattr(hub_constants, name):
                setattr(hub_constants, name, str(hub_cache))
        if hasattr(hub_constants, "HF_HOME"):
            setattr(hub_constants, "HF_HOME", str(cache_path))

    datasets_config = sys.modules.get("datasets.config")
    if datasets_config is not None and hasattr(datasets_config, "HF_DATASETS_CACHE"):
        datasets_config.HF_DATASETS_CACHE = str(datasets_cache)

    return cache_path


def prepare_optional_ml_dependencies() -> None:
    """Ignore broken optional multimedia libraries for this text-only pipeline.

    Transformers detects optional libraries by package metadata. Stale native
    wheels can therefore be considered available even when their DLLs are
    incompatible with the installed PyTorch build. GPT-OSS does not use audio,
    image, or video processing, so those optional packages can safely be marked
    unavailable when importing them fails.
    """
    global _OPTIONAL_DEPENDENCIES_PREPARED
    if _OPTIONAL_DEPENDENCIES_PREPARED:
        return
    _OPTIONAL_DEPENDENCIES_PREPARED = True

    warnings.filterwarnings(
        "ignore",
        message=r"urllib3 .* or chardet .* doesn't match a supported version!",
        module=r"requests(\..*)?",
    )

    for package_name, availability_name in (
        ("torchaudio", "is_torchaudio_available"),
        ("torchvision", "is_torchvision_available"),
        ("librosa", "is_librosa_available"),
    ):
        try:
            importlib.import_module(package_name)
        except ModuleNotFoundError as error:
            if error.name == package_name:
                continue
            _disable_transformers_optional(package_name, availability_name, error)
        except (ImportError, OSError, RuntimeError, AttributeError) as error:
            _disable_transformers_optional(package_name, availability_name, error)


def _disable_transformers_optional(
    package_name: str,
    availability_name: str,
    error: BaseException,
) -> None:
    logger.warning(
        "Ignoring broken optional %s installation for text-model evaluation: %s. "
        "Reinstall it from the same PyTorch index/version if multimedia support is required.",
        package_name,
        error,
    )
    try:
        import transformers
        from transformers.utils import import_utils

        def unavailable() -> bool:
            return False

        availability_function = getattr(import_utils, availability_name)
        clear_cache = getattr(availability_function, "cache_clear", None)
        if clear_cache:
            clear_cache()
        setattr(import_utils, availability_name, unavailable)
        setattr(transformers.utils, availability_name, unavailable)
        setattr(transformers, availability_name, unavailable)
    except (ImportError, AttributeError):
        # Best-effort: Transformers may not be installed during lightweight tests.
        pass


def free_torch_memory() -> None:
    """Drop unused CUDA/RAM so a Abliteration child can map the model on Windows.

    Must not call ``torch.cuda.is_available()`` unless CUDA is already
    initialized. That call creates a parent-process CUDA context (~0.5-1 GB)
    which then steals VRAM from the Abliteration child and 4-bit 9B loads crash.
    """
    import gc

    gc.collect()
    try:
        import torch

        initialized = getattr(torch.cuda, "is_initialized", None)
        if callable(initialized):
            if not initialized():
                return
        elif not torch.cuda.is_available():
            return
        torch.cuda.empty_cache()
        ipc_collect = getattr(torch.cuda, "ipc_collect", None)
        if callable(ipc_collect):
            ipc_collect()
        synchronize = getattr(torch.cuda, "synchronize", None)
        if callable(synchronize):
            synchronize()
    except Exception as error:
        logger.debug("Could not free CUDA memory: %s", error)
    gc.collect()
