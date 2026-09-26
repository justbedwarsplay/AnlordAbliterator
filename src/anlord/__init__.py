# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Anlord Abliterator

Automated LLM abliteration followed by evaluation,
benchmarking, comparison, and report generation.
"""

import os

__author__ = "Anlord Abliterator Development Team"

# Release constant, kept in sync with pyproject.toml on every release.
_FALLBACK_VERSION = "1.4.0"

# The reported version must match how the code is actually running:
# - installed wheel (site-packages) -> read the pip distribution metadata,
#   which is always the version the user installed;
# - source checkout (PYTHONPATH=src / python -m src.anlord) -> the release
#   constant above, because a stale installed distribution's metadata would
#   lie about the code being executed.
_HERE = os.path.dirname(__file__).replace("\\", "/")
_RUNNING_FROM_WHEEL = "site-packages" in _HERE or "dist-packages" in _HERE

if _RUNNING_FROM_WHEEL:
    try:
        from importlib.metadata import version as _metadata_version

        for _distribution in ("AnlordAbliterator", "anlordabliterator", "anlord"):
            try:
                __version__ = _metadata_version(_distribution)
                break
            except Exception:
                continue
        else:
            __version__ = _FALLBACK_VERSION
    except Exception:
        __version__ = _FALLBACK_VERSION
else:
    __version__ = _FALLBACK_VERSION

# Note: Use lazy imports to avoid importing torch before it's installed
# from anlord.config import Settings, BenchmarkConfig, HardwareMetrics, RunMode
# from anlord.pipeline import AbliterationPipeline

__all__ = [
    "__version__",
]
