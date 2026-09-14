# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Evaluation module for Anlord Abliterator.
Handles baseline evaluation and model comparison.
"""

from .evaluator import (
    BaselineEvaluator,
    EvaluationResult,
    compare_models,
)
from .comparison import (
    ComparisonResult,
    compare_results,
    generate_delta_table,
)

__all__ = [
    "BaselineEvaluator",
    "EvaluationResult",
    "compare_models",
    "ComparisonResult",
    "compare_results",
    "generate_delta_table",
]
