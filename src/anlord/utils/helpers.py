# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Helper utilities for Anlord Abliterator.
"""

from pathlib import Path
from typing import Optional


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted string (e.g., "01:30:45")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def format_percentage(value: float, decimals: int = 1) -> str:
    """
    Format a value as a percentage string.

    Args:
        value: Value between 0 and 1 (or already a percentage)
        decimals: Number of decimal places

    Returns:
        Formatted percentage string
    """
    # If value is > 1, assume it's already a percentage
    if abs(value) > 1:
        return f"{value:.{decimals}f}%"

    return f"{value * 100:.{decimals}f}%"


def ensure_dir(path: Path | str) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path

    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.

    Args:
        numerator: The numerator
        denominator: The denominator
        default: Default value if denominator is zero

    Returns:
        Result of division or default
    """
    if denominator == 0:
        return default
    return numerator / denominator


def format_delta(delta: float, relative: Optional[float] = None) -> str:
    """
    Format a delta value for display.

    Args:
        delta: Absolute delta value
        relative: Optional relative delta (percentage)

    Returns:
        Formatted string
    """
    sign = "+" if delta >= 0 else ""
    result = f"{sign}{delta:.4f}"

    if relative is not None:
        result += f" ({sign}{relative:.2f}%)"

    return result


def truncate_string(s: str, max_length: int = 50, suffix: str = "...") -> str:
    """
    Truncate a string to maximum length.

    Args:
        s: String to truncate
        max_length: Maximum length
        suffix: Suffix to append if truncated

    Returns:
        Truncated string
    """
    if len(s) <= max_length:
        return s

    return s[: max_length - len(suffix)] + suffix
