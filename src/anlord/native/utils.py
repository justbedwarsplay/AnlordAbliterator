# SPDX-License-Identifier: AGPL-3.0-or-later
"""Native utilities — mirrors abliteration/system.py and abliteration/utils.py helpers."""

from __future__ import annotations

import os
import random
import gc
import torch
import numpy as np


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
