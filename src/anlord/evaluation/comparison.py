# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Model comparison module for Anlord Abliterator.
Compares baseline and abliterated model performance.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime

from .evaluator import EvaluationResult

logger = logging.getLogger(__name__)


def format_optional(value, spec: str = "{}") -> str:
    """Renders an optional metric: "n/a" when unknown, formatted otherwise."""
    return "n/a" if value is None else spec.format(value)


@dataclass
class ComparisonResult:
    """Results from comparing baseline and abliterated models."""

    model_id: str
    timestamp: str = ""

    # Refusal metrics (None = not measured, rendered as "n/a")
    initial_refusals_baseline: Optional[int] = None
    initial_refusals_abliterated: Optional[int] = None
    final_refusals_baseline: Optional[int] = None
    final_refusals_abliterated: Optional[int] = None
    kl_divergence: Optional[float] = None

    # Benchmark comparison
    benchmarks: dict = field(default_factory=dict)

    # Summary statistics
    total_duration_seconds: float = 0.0
    peak_vram_gb: float = 0.0
    peak_ram_gb: float = 0.0

    # Warnings and notes
    warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "model_id": self.model_id,
            "timestamp": self.timestamp,
            "initial_refusals_baseline": self.initial_refusals_baseline,
            "initial_refusals_abliterated": self.initial_refusals_abliterated,
            "final_refusals_baseline": self.final_refusals_baseline,
            "final_refusals_abliterated": self.final_refusals_abliterated,
            "kl_divergence": self.kl_divergence,
            "benchmarks": self.benchmarks,
            "total_duration_seconds": self.total_duration_seconds,
            "peak_vram_gb": self.peak_vram_gb,
            "peak_ram_gb": self.peak_ram_gb,
            "warnings": self.warnings,
        }

    @property
    def refusal_reduction(self) -> Optional[int]:
        """Total reduction in refusals (None when not measured)."""
        if self.final_refusals_baseline is None or self.final_refusals_abliterated is None:
            return None
        return self.final_refusals_baseline - self.final_refusals_abliterated

    @property
    def refusal_reduction_percent(self) -> Optional[float]:
        """Percentage reduction in refusals (None when not measured)."""
        reduction = self.refusal_reduction
        if reduction is None:
            return None
        if not self.final_refusals_baseline:
            return 0.0
        return (reduction / self.final_refusals_baseline) * 100


def compare_results(
    baseline: EvaluationResult,
    abliterated: EvaluationResult,
) -> ComparisonResult:
    """
    Compare baseline and abliterated evaluation results.

    Args:
        baseline: Baseline evaluation results
        abliterated: Abliterated evaluation results

    Returns:
        ComparisonResult with all comparison metrics
    """
    comparison = ComparisonResult(
        model_id=baseline.model_id,
        timestamp=datetime.now().isoformat(),
    )

    # Extract refusal metrics
    if baseline.abliteration:
        comparison.initial_refusals_baseline = baseline.abliteration.initial_refusals
        comparison.final_refusals_baseline = baseline.abliteration.final_refusals

    if abliterated.abliteration:
        comparison.initial_refusals_abliterated = abliterated.abliteration.initial_refusals
        comparison.final_refusals_abliterated = abliterated.abliteration.final_refusals
        comparison.kl_divergence = abliterated.abliteration.kl_divergence

    # Compare benchmarks
    common_tasks = set(baseline.benchmarks.keys()) & set(abliterated.benchmarks.keys())

    for task_id in common_tasks:
        baseline_result = baseline.benchmarks[task_id]
        abliterated_result = abliterated.benchmarks[task_id]

        if baseline_result.error or abliterated_result.error:
            comparison.warnings.append(
                f"{baseline_result.task_name} was excluded because evaluation failed: "
                f"{baseline_result.error or abliterated_result.error}"
            )
            continue

        baseline_score = baseline_result.primary_metric
        abliterated_score = abliterated_result.primary_metric

        absolute_delta = abliterated_score - baseline_score
        relative_delta = (absolute_delta / baseline_score * 100) if baseline_score != 0 else 0.0

        comparison.benchmarks[task_id] = {
            "name": baseline_result.task_name,
            "baseline": baseline_score,
            "baseline_stderr": baseline_result.primary_metric_stderr,
            "abliterated": abliterated_score,
            "abliterated_stderr": abliterated_result.primary_metric_stderr,
            "absolute_delta": absolute_delta,
            "relative_delta_percent": relative_delta,
            "baseline_samples": baseline_result.num_samples,
            "abliterated_samples": abliterated_result.num_samples,
            "error": (abliterated_result.error if abliterated_result.error else None),
        }

    # Aggregate timing
    comparison.total_duration_seconds = baseline.duration_seconds + abliterated.duration_seconds

    # Peak memory
    comparison.peak_vram_gb = max(
        baseline.peak_vram_gb,
        abliterated.peak_vram_gb,
    )
    comparison.peak_ram_gb = max(
        baseline.peak_ram_gb,
        abliterated.peak_ram_gb,
    )

    # Add warnings
    if comparison.kl_divergence is not None and comparison.kl_divergence > 1.0:
        comparison.warnings.append(
            f"KL divergence ({comparison.kl_divergence:.3f}) is relatively high. "
            "The abliterated model may have diverged significantly from the original."
        )

    # Check for significant benchmark drops
    for task_id, metrics in comparison.benchmarks.items():
        if metrics["relative_delta_percent"] < -5.0:
            comparison.warnings.append(
                f"{metrics['name']} shows significant degradation "
                f"({metrics['relative_delta_percent']:.1f}%)."
            )

    return comparison


def generate_delta_table(comparison: ComparisonResult) -> str:
    """
    Generate a formatted delta table for display.

    Args:
        comparison: Comparison results

    Returns:
        Formatted string table
    """
    lines = []

    # Header
    lines.append("")
    lines.append("=" * 80)
    lines.append("BENCHMARK COMPARISON TABLE")
    lines.append("=" * 80)
    lines.append("")

    # Column headers
    header = f"{'Benchmark':<20} {'Original':<12} {'Abliterated':<12} {'Delta':<12} {'Rel. %':<10}"
    lines.append(header)
    lines.append("-" * 80)

    # Data rows
    for task_id, metrics in sorted(comparison.benchmarks.items()):
        name = metrics["name"][:19]
        baseline = f"{metrics['baseline']:.4f}"
        abliterated = f"{metrics['abliterated']:.4f}"

        delta = metrics["absolute_delta"]
        delta_str = f"{delta:+.4f}"

        rel_delta = metrics["relative_delta_percent"]
        rel_str = f"{rel_delta:+.2f}%"

        if delta < 0:
            delta_str = f"\033[91m{delta_str}\033[0m"  # Red
        elif delta > 0:
            delta_str = f"\033[92m{delta_str}\033[0m"  # Green

        lines.append(f"{name:<20} {baseline:<12} {abliterated:<12} {delta_str:<12} {rel_str:<10}")

    lines.append("-" * 80)
    lines.append("")

    # Summary
    lines.append("REFUSAL METRICS:")
    lines.append(f"  Initial refusals (baseline):    {comparison.initial_refusals_baseline}")
    lines.append(f"  Initial refusals (abliterated): {comparison.initial_refusals_abliterated}")
    lines.append(f"  Final refusals (baseline):      {comparison.final_refusals_baseline}")
    lines.append(f"  Final refusals (abliterated):   {comparison.final_refusals_abliterated}")
    lines.append(f"  KL divergence:                   {comparison.kl_divergence:.6f}")
    lines.append("")

    lines.append("NOTES:")
    lines.append("  - KL divergence is NOT a percentage of retained quality.")
    lines.append("  - Delta shows absolute change in benchmark scores.")
    lines.append("  - Relative % shows percentage change relative to baseline.")
    lines.append("")

    if comparison.warnings:
        lines.append("WARNINGS:")
        for warning in comparison.warnings:
            lines.append(f"  ⚠ {warning}")
        lines.append("")

    lines.append("=" * 80)

    return "\n".join(lines)


def format_comparison_for_markdown(comparison: ComparisonResult) -> str:
    """
    Format comparison results as Markdown.

    Args:
        comparison: Comparison results

    Returns:
        Markdown formatted string
    """
    lines = []

    # Header
    lines.append(f"# Abliteration Results for {comparison.model_id}")
    lines.append("")
    lines.append(f"**Generated:** {comparison.timestamp}")
    lines.append("")

    # Refusal Metrics
    lines.append("## Refusal Metrics")
    lines.append("")
    lines.append("| Metric | Baseline | Abliterated | Change |")
    lines.append("|--------|----------|-------------|--------|")

    baseline_refusals = comparison.final_refusals_baseline
    abliterated_refusals = comparison.final_refusals_abliterated
    change = abliterated_refusals - baseline_refusals

    lines.append(f"| Final Refusals | {baseline_refusals} | {abliterated_refusals} | {change:+d} |")
    lines.append(f"| KL Divergence | - | {comparison.kl_divergence:.6f} | - |")
    lines.append("")

    # Benchmark Comparison
    lines.append("## Benchmark Comparison")
    lines.append("")
    lines.append("| Benchmark | Original | Abliterated | Delta | Relative |")
    lines.append("|-----------|----------|-------------|-------|----------|")

    for task_id, metrics in sorted(comparison.benchmarks.items()):
        name = metrics["name"]
        baseline = f"{metrics['baseline']:.4f}"
        abliterated = f"{metrics['abliterated']:.4f}"
        delta = f"{metrics['absolute_delta']:+.4f}"
        rel = f"{metrics['relative_delta_percent']:+.2f}%"

        lines.append(f"| {name} | {baseline} | {abliterated} | {delta} | {rel} |")

    lines.append("")

    # Important Notes
    lines.append("## Important Notes")
    lines.append("")
    lines.append("1. **KL divergence is NOT a percentage of retained quality.**")
    lines.append("   It measures the statistical distance between the original and")
    lines.append("   abliterated model distributions.")
    lines.append("")
    lines.append("2. **Benchmark changes alone do not determine model quality.**")
    lines.append("   Small benchmark variations (within statistical margin) are expected.")
    lines.append("")
    lines.append("3. **Behavior change vs. capability change:**")
    lines.append("   - Refusal reduction indicates reduced safety behavior")
    lines.append("   - KL divergence indicates model drift")
    lines.append("   - Benchmark changes indicate capability modification")
    lines.append("")

    # Warnings
    if comparison.warnings:
        lines.append("## Warnings")
        lines.append("")
        for warning in comparison.warnings:
            lines.append(f"- ⚠ {warning}")
        lines.append("")

    # Hardware Summary
    lines.append("## Hardware Summary")
    lines.append("")
    lines.append(f"- Peak VRAM: {comparison.peak_vram_gb:.2f} GB")
    lines.append(f"- Peak RAM: {comparison.peak_ram_gb:.2f} GB")
    lines.append(f"- Total Duration: {comparison.total_duration_seconds:.1f} seconds")
    lines.append("")

    return "\n".join(lines)
