# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Benchmarks module for Anlord Abliterator.
Provides integration with lm-evaluation-harness and native implementations.
"""

from .runner import BenchmarkResult, BenchmarkRunner

try:
    from .native import NativeBenchmarkRunner
except ImportError:
    NativeBenchmarkRunner = None  # type: ignore

from .tasks import AVAILABLE_TASKS, list_available_tasks

__all__ = [
    "BenchmarkRunner",
    "BenchmarkResult",
    "NativeBenchmarkRunner",
    "AVAILABLE_TASKS",
    "list_available_tasks",
]
