# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Heretic integration module for Anlord Abliterator.
Wraps Heretic functionality for use in the pipeline.
"""

from .wrapper import HereticWrapper, HereticResult
from .compatibility import check_model_compatibility, SUPPORTED_ARCHITECTURES

__all__ = [
    "HereticWrapper",
    "HereticResult",
    "check_model_compatibility",
    "SUPPORTED_ARCHITECTURES",
]
