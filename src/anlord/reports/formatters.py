# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Formatters for Anlord Abliterator reports.
Provides various output format utilities.
"""

import json

from ..evaluation.comparison import ComparisonResult


def format_json(comparison: ComparisonResult, indent: int = 2) -> str:
    """
    Format comparison as JSON string.

    Args:
        comparison: Comparison results
        indent: JSON indentation

    Returns:
        JSON formatted string
    """
    return json.dumps(comparison.to_dict(), indent=indent)


def format_csv(comparison: ComparisonResult) -> str:
    """
    Format comparison as CSV string.

    Args:
        comparison: Comparison results

    Returns:
        CSV formatted string
    """
    lines = []

    # Header
    lines.append("Anlord Abliterator Report")
    lines.append(f"Model,{comparison.model_id}")
    lines.append(f"Generated,{comparison.timestamp}")
    lines.append("")

    # Refusal metrics
    lines.append("Refusal Metrics")
    lines.append("Metric,Baseline,Abliterated,Change")
    lines.append(
        f"Initial Refusals,{comparison.initial_refusals_baseline},{comparison.initial_refusals_abliterated},{comparison.initial_refusals_abliterated - comparison.initial_refusals_baseline}"
    )
    lines.append(
        f"Final Refusals,{comparison.final_refusals_baseline},{comparison.final_refusals_abliterated},{comparison.final_refusals_abliterated - comparison.final_refusals_baseline}"
    )
    lines.append(f"KL Divergence,-,{comparison.kl_divergence:.6f},-")
    lines.append("")

    # Benchmarks
    lines.append("Benchmark Comparison")
    lines.append("Benchmark,Original,Abliterated,Delta,Relative %")
    for task_id, metrics in sorted(comparison.benchmarks.items()):
        lines.append(
            f"{metrics['name']},{metrics['baseline']:.4f},{metrics['abliterated']:.4f},"
            f"{metrics['absolute_delta']:+.4f},{metrics['relative_delta_percent']:+.2f}%"
        )
    lines.append("")

    # Hardware
    lines.append("Hardware Summary")
    lines.append("Metric,Value")
    lines.append(f"Peak VRAM (GB),{comparison.peak_vram_gb:.2f}")
    lines.append(f"Peak RAM (GB),{comparison.peak_ram_gb:.2f}")
    lines.append(f"Duration (s),{comparison.total_duration_seconds:.1f}")

    return "\n".join(lines)


def format_html(comparison: ComparisonResult, theme: str = "dark") -> str:
    """
    Format comparison as HTML string.

    Args:
        comparison: Comparison results
        theme: Color theme ("dark" or "light")

    Returns:
        HTML formatted string
    """
    from .generator import ReportGenerator

    generator = ReportGenerator(output_dir=".")
    return generator._build_html(comparison=comparison)


def format_markdown(comparison: ComparisonResult) -> str:
    """
    Format comparison as Markdown string.

    Args:
        comparison: Comparison results

    Returns:
        Markdown formatted string
    """
    from ..evaluation.comparison import format_comparison_for_markdown

    return format_comparison_for_markdown(comparison)


def format_table(comparison: ComparisonResult) -> str:
    """
    Format comparison as ASCII table.

    Args:
        comparison: Comparison results

    Returns:
        ASCII table string
    """
    from ..evaluation.comparison import generate_delta_table

    return generate_delta_table(comparison)
