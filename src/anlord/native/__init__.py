# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Anlord native abliterator — 1:1 replication of p-e-w/abliteration inside AnlordAbliratorDev.

This package provides a self-contained implementation of the Abliteration abliteration
pipeline without importing `abliteration-llm` at runtime. Every stage mirrors the
original Abliteration source (residual extraction, refusal directions, LoRA updates,
KL divergence, Optuna/TPE optimization) so that numerical parity can be verified.

Public entry point: NativeAbliterator
"""

from .abliterator import NativeAbliterator, NativeAbliteratorResult
from .config import NativeConfig

__all__ = ["NativeAbliterator", "NativeAbliteratorResult", "NativeConfig"]
