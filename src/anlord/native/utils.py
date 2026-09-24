# SPDX-License-Identifier: AGPL-3.0-or-later
"""Native utilities — mirrors abliteration/system.py and abliteration/utils.py helpers."""

from __future__ import annotations

import hashlib
import os
import random
import gc
import traceback
import torch
import numpy as np
from optuna.study import StudyDirection
from rich.console import Console

print = Console(highlight=False).print


def deep_merge_dicts(base: dict, override: dict) -> dict:
    """
    Recursively merge two dicts.

    Values from `override` take precedence. Nested dicts are merged recursively.
    """
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def parse_study_direction(optimization: str) -> StudyDirection:
    """
    Converts the optimization value stored as a `str` to the
    `StudyDirection` object required by Optuna.
    """
    if optimization == "none":
        return StudyDirection.NOT_SET
    return StudyDirection[optimization.upper()]


def get_file_sha256(file_path: str | os.PathLike) -> str:
    """Computes the SHA-256 hash of a file, reading it in 64 kB blocks."""
    hash = hashlib.sha256()

    with open(file_path, "rb") as file:
        for block in iter(lambda: file.read(65536), b""):
            hash.update(block)

    return hash.hexdigest()


def print_memory_usage() -> None:
    def p(label: str, size_in_bytes: int):
        print(f"[grey50]{label}: [bold]{size_in_bytes / (1024**3):.2f} GB[/][/]")

    try:
        from psutil import Process

        p("Resident system RAM", Process().memory_info().rss)
    except Exception:
        pass

    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        allocated = sum(torch.cuda.memory_allocated(device) for device in range(count))
        reserved = sum(torch.cuda.memory_reserved(device) for device in range(count))
        p("Allocated GPU VRAM", allocated)
        p("Reserved GPU VRAM", reserved)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        p("Allocated MPS memory", torch.mps.current_allocated_memory())
        p("Driver (reserved) MPS memory", torch.mps.driver_allocated_memory())


def format_exception(error: Exception) -> str:
    # Walk causal chain to find a non-empty message.
    current = error
    while current is not None:
        message = str(current).strip()
        if message:
            return message
        current = current.__cause__ or current.__context__

    # If there is no message in the entire causal chain, fall back to the complete traceback.
    return traceback.format_exc().strip()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def empty_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "xpu") and torch.xpu.is_available():  # type: ignore[attr-defined]
        torch.xpu.empty_cache()  # type: ignore[attr-defined]
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()  # type: ignore[attr-defined]


def format_duration(seconds: float) -> str:
    seconds = round(seconds)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if hours > 0:
        return f"{hours}h {minutes}m"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"


def get_trial_parameters(trial) -> dict:
    params = dict(trial.params)
    # pretty
    out: dict[str, str] = {}
    for k, v in params.items():
        if isinstance(v, float):
            out[k] = f"{v:.4f}"
        else:
            out[k] = str(v)
    return out
