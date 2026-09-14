# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Comparison module re-export for convenience.
"""

from ..evaluation.comparison import (
    ComparisonResult,
    compare_results,
    generate_delta_table,
    format_comparison_for_markdown,
)

__all__ = [
    "ComparisonResult",
    "compare_results",
    "generate_delta_table",
    "format_comparison_for_markdown",
]
