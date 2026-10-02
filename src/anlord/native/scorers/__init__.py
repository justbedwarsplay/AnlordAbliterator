# SPDX-License-Identifier: AGPL-3.0-or-later
"""Built-in scorer plugins."""

from .keyword_rate import KeywordRate
from .kl_divergence import KLDivergence

try:
    from .benchmark_score import BenchmarkScore
except ImportError:  # pragma: no cover - lm_eval is an optional extra ('lmeval')
    BenchmarkScore = None

__all__ = ["KeywordRate", "KLDivergence", "BenchmarkScore"]
