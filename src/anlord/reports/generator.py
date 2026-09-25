# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Report generator for Anlord Abliterator.
Creates HTML, JSON, and CSV reports from comparison results.
"""

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..evaluation.comparison import ComparisonResult
from ..evaluation.comparison import format_optional as fmt
from ..evaluation.evaluator import EvaluationResult

logger = logging.getLogger(__name__)


@dataclass
class ReportConfig:
    """Configuration for report generation."""

    output_dir: Path
    include_raw_data: bool = True
    include_warnings: bool = True
    include_system_info: bool = True
    theme: str = "dark"  # dark or light


class ReportGenerator:
    """
    Generator for Anlord Abliterator reports.

    Creates comprehensive HTML, JSON, and CSV reports
    from comparison results.
    """

    def __init__(
        self,
        output_dir: Path | str,
        config: Optional[ReportConfig] = None,
    ):
        """
        Initialize report generator.

        Args:
            output_dir: Directory for output reports
            config: Report configuration
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or ReportConfig(output_dir=self.output_dir)

    def generate_html_report(
        self,
        comparison: ComparisonResult,
        baseline: Optional[EvaluationResult] = None,
        abliterated: Optional[EvaluationResult] = None,
        filename: str = "report.html",
    ) -> Path:
        """
        Generate HTML report.

        Args:
            comparison: Comparison results
            baseline: Optional baseline evaluation
            abliterated: Optional abliterated evaluation
            filename: Output filename

        Returns:
            Path to generated report
        """
        filepath = self.output_dir / filename

        html_content = self._build_html(
            comparison=comparison,
            baseline=baseline,
            abliterated=abliterated,
        )

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info(f"Generated HTML report: {filepath}")
        return filepath

    def generate_json_report(
        self,
        comparison: ComparisonResult,
        filename: str = "report.json",
    ) -> Path:
        """
        Generate JSON report.

        Args:
            comparison: Comparison results
            filename: Output filename

        Returns:
            Path to generated report
        """
        filepath = self.output_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(comparison.to_dict(), f, indent=2)

        logger.info(f"Generated JSON report: {filepath}")
        return filepath

    def generate_csv_report(
        self,
        comparison: ComparisonResult,
        filename: str = "report.csv",
    ) -> Path:
        """
        Generate CSV report.

        Args:
            comparison: Comparison results
            filename: Output filename

        Returns:
            Path to generated report
        """
        filepath = self.output_dir / filename

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # Header section
            writer.writerow(["Anlord Abliterator Report"])
            writer.writerow(["Model", comparison.model_id])
            writer.writerow(["Generated", comparison.timestamp])
            writer.writerow([])

            # Refusal metrics
            writer.writerow(["Refusal Metrics"])
            writer.writerow(["Metric", "Baseline", "Abliterated", "Change"])
            writer.writerow(
                [
                    "Initial Refusals",
                    fmt(comparison.initial_refusals_baseline),
                    fmt(comparison.initial_refusals_abliterated),
                    fmt(
                        None
                        if comparison.initial_refusals_baseline is None
                        or comparison.initial_refusals_abliterated is None
                        else comparison.initial_refusals_abliterated - comparison.initial_refusals_baseline
                    ),
                ]
            )
            writer.writerow(
                [
                    "Final Refusals",
                    fmt(comparison.final_refusals_baseline),
                    fmt(comparison.final_refusals_abliterated),
                    fmt(
                        None
                        if comparison.final_refusals_baseline is None
                        or comparison.final_refusals_abliterated is None
                        else comparison.final_refusals_abliterated - comparison.final_refusals_baseline
                    ),
                ]
            )
            writer.writerow(["KL Divergence", "-", fmt(comparison.kl_divergence, "{:.6f}"), "-"])
            writer.writerow([])

            # Benchmark comparison
            writer.writerow(["Benchmark Comparison"])
            writer.writerow(
                [
                    "Benchmark",
                    "Original",
                    "Abliterated",
                    "Delta",
                    "Relative %",
                ]
            )

            for task_id, metrics in sorted(comparison.benchmarks.items()):
                writer.writerow(
                    [
                        metrics["name"],
                        f"{metrics['baseline']:.4f}",
                        f"{metrics['abliterated']:.4f}",
                        f"{metrics['absolute_delta']:+.4f}",
                        f"{metrics['relative_delta_percent']:+.2f}%",
                    ]
                )

            writer.writerow([])

            # Hardware summary
            writer.writerow(["Hardware Summary"])
            writer.writerow(["Metric", "Value"])
            writer.writerow(["Peak VRAM (GB)", f"{comparison.peak_vram_gb:.2f}"])
            writer.writerow(["Peak RAM (GB)", f"{comparison.peak_ram_gb:.2f}"])
            writer.writerow(["Duration (s)", f"{comparison.total_duration_seconds:.1f}"])

            # Warnings
            if comparison.warnings and self.config.include_warnings:
                writer.writerow([])
                writer.writerow(["Warnings"])
                for warning in comparison.warnings:
                    writer.writerow([warning])

        logger.info(f"Generated CSV report: {filepath}")
        return filepath

    def _build_html(
        self,
        comparison: ComparisonResult,
        baseline: Optional[EvaluationResult] = None,
        abliterated: Optional[EvaluationResult] = None,
    ) -> str:
        """Build HTML content for the report."""

        # Build benchmark rows
        benchmark_rows = ""
        for task_id, metrics in sorted(comparison.benchmarks.items()):
            delta = metrics["absolute_delta"]
            delta_class = "positive" if delta > 0 else "negative" if delta < 0 else "neutral"

            benchmark_rows += f"""
            <tr>
                <td>{metrics["name"]}</td>
                <td>{metrics["baseline"]:.4f}</td>
                <td>{metrics["abliterated"]:.4f}</td>
                <td class="{delta_class}">{delta:+.4f}</td>
                <td class="{delta_class}">{metrics["relative_delta_percent"]:+.2f}%</td>
            </tr>
            """

        # Build warnings
        warnings_html = ""
        if comparison.warnings:
            warnings_html = '<div class="warnings"><h3>⚠ Warnings</h3><ul>'
            for warning in comparison.warnings:
                warnings_html += f"<li>{warning}</li>"
            warnings_html += "</ul></div>"

        # Build comparison delta table
        delta_table_rows = ""
        for task_id, metrics in sorted(comparison.benchmarks.items()):
            delta = metrics["absolute_delta"]
            delta_table_rows += f"""
                <tr>
                    <td>{metrics["name"]}</td>
                    <td>{metrics["baseline"]:.4f}</td>
                    <td>{metrics["abliterated"]:.4f}</td>
                    <td style="color: {"#4CAF50" if delta > 0 else "#f44336" if delta < 0 else "inherit"}">{delta:+.4f}</td>
                </tr>
            """

        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Anlord Abliterator Report - {comparison.model_id}</title>
    <style>
        :root {{
            --bg-primary: #0f0f0f;
            --bg-secondary: #1a1a1a;
            --bg-tertiary: #252525;
            --text-primary: #ffffff;
            --text-secondary: #b0b0b0;
            --accent: #00d4ff;
            --accent-secondary: #7c4dff;
            --positive: #4CAF50;
            --negative: #f44336;
            --border: #333333;
        }}
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 2rem;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        
        header {{
            text-align: center;
            margin-bottom: 3rem;
            padding: 2rem;
            background: var(--bg-secondary);
            border-radius: 12px;
            border: 1px solid var(--border);
        }}
        
        h1 {{
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
            background: linear-gradient(135deg, var(--accent), var(--accent-secondary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        
        .subtitle {{
            color: var(--text-secondary);
            font-size: 1.1rem;
        }}
        
        .section {{
            background: var(--bg-secondary);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            border: 1px solid var(--border);
        }}
        
        h2 {{
            font-size: 1.5rem;
            margin-bottom: 1rem;
            color: var(--accent);
            border-bottom: 2px solid var(--accent);
            padding-bottom: 0.5rem;
        }}
        
        h3 {{
            font-size: 1.2rem;
            margin: 1rem 0 0.5rem;
            color: var(--text-secondary);
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
        }}
        
        th, td {{
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}
        
        th {{
            background: var(--bg-tertiary);
            color: var(--text-secondary);
            font-weight: 600;
        }}
        
        tr:hover {{
            background: var(--bg-tertiary);
        }}
        
        .positive {{ color: var(--positive); }}
        .negative {{ color: var(--negative); }}
        .neutral {{ color: var(--text-secondary); }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin: 1rem 0;
        }}
        
        .metric-card {{
            background: var(--bg-tertiary);
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
        }}
        
        .metric-value {{
            font-size: 2rem;
            font-weight: bold;
            color: var(--accent);
        }}
        
        .metric-label {{
            color: var(--text-secondary);
            font-size: 0.9rem;
        }}
        
        .warnings {{
            background: rgba(244, 67, 54, 0.1);
            border: 1px solid var(--negative);
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
        }}
        
        .warnings h3 {{
            color: var(--negative);
            margin-top: 0;
        }}
        
        .warnings ul {{
            margin-left: 1.5rem;
            color: var(--text-secondary);
        }}
        
        .info-box {{
            background: rgba(0, 212, 255, 0.1);
            border: 1px solid var(--accent);
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
        }}
        
        .info-box p {{
            margin: 0.5rem 0;
            color: var(--text-secondary);
        }}
        
        .info-box strong {{
            color: var(--text-primary);
        }}
        
        footer {{
            text-align: center;
            margin-top: 2rem;
            padding: 1rem;
            color: var(--text-secondary);
            font-size: 0.9rem;
        }}
        
        .delta-table {{
            margin: 1rem 0;
        }}
        
        .delta-table table {{
            margin: 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Anlord Abliterator Report</h1>
            <p class="subtitle">Model: {comparison.model_id}</p>
            <p class="subtitle">Generated: {comparison.timestamp}</p>
        </header>
        
        <section class="section">
            <h2>📊 Summary</h2>
            
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">{fmt(comparison.refusal_reduction_percent, "{:.1f}%")}</div>
                    <div class="metric-label">Refusal Reduction</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{fmt(comparison.final_refusals_abliterated)}</div>
                    <div class="metric-label">Final Refusals</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{fmt(comparison.kl_divergence, "{:.4f}")}</div>
                    <div class="metric-label">KL Divergence</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{len(comparison.benchmarks)}</div>
                    <div class="metric-label">Benchmarks</div>
                </div>
            </div>
        </section>
        
        <section class="section">
            <h2>🚫 Refusal Metrics</h2>
            
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">{fmt(comparison.initial_refusals_baseline)}</div>
                    <div class="metric-label">Initial Refusals (Baseline)</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{fmt(comparison.initial_refusals_abliterated)}</div>
                    <div class="metric-label">Initial Refusals (Abliterated)</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{fmt(comparison.final_refusals_baseline)}</div>
                    <div class="metric-label">Final Refusals (Baseline)</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{comparison.final_refusals_abliterated}</div>
                    <div class="metric-label">Final Refusals (Abliterated)</div>
                </div>
            </div>
            
            <div class="info-box">
                <p><strong>KL Divergence: {fmt(comparison.kl_divergence, "{:.6f}")}</strong></p>
                <p><strong>Important:</strong> KL divergence is NOT a percentage of retained quality. It measures the statistical distance between the original and abliterated model distributions. A lower KL divergence indicates less model drift, but does not directly correlate with capability preservation.</p>
            </div>
        </section>
        
        <section class="section">
            <h2>📈 Benchmark Comparison</h2>
            
            <table>
                <thead>
                    <tr>
                        <th>Benchmark</th>
                        <th>Original</th>
                        <th>Abliterated</th>
                        <th>Delta</th>
                        <th>Relative %</th>
                    </tr>
                </thead>
                <tbody>
                    {benchmark_rows}
                </tbody>
            </table>
        </section>
        
        {warnings_html}
        
        <section class="section">
            <h2>📋 Comparison Table</h2>
            
            <div class="delta-table">
                <table>
                    <thead>
                        <tr>
                            <th>Benchmark</th>
                            <th>Original</th>
                            <th>Abliterated</th>
                            <th>Delta</th>
                        </tr>
                    </thead>
                    <tbody>
                        {delta_table_rows}
                    </tbody>
                </table>
            </div>
        </section>
        
        <section class="section">
            <h2>💻 Hardware Summary</h2>
            
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">{comparison.peak_vram_gb:.1f} GB</div>
                    <div class="metric-label">Peak VRAM</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{comparison.peak_ram_gb:.1f} GB</div>
                    <div class="metric-label">Peak RAM</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{comparison.total_duration_seconds / 60:.1f} min</div>
                    <div class="metric-label">Total Duration</div>
                </div>
            </div>
        </section>
        
        <section class="section">
            <h2>📝 Important Notes</h2>
            
            <div class="info-box">
                <h3>Understanding Abliteration Impact</h3>
                <p><strong>Behavior Change:</strong> The reduction in refusal rate indicates the model has been modified to be less likely to refuse potentially sensitive queries.</p>
                <p><strong>Model Drift:</strong> KL divergence measures how much the model's probability distributions have changed from the original.</p>
                <p><strong>Capability Change:</strong> Benchmark changes indicate modifications to the model's general capabilities.</p>
            </div>
        </section>
        
        <footer>
            <p>Generated by Anlord Abliterator</p>
        </footer>
    </div>
</body>
</html>
        """

        return html

    def generate_all(
        self,
        comparison: ComparisonResult,
        baseline: Optional[EvaluationResult] = None,
        abliterated: Optional[EvaluationResult] = None,
    ) -> dict[str, Path]:
        """
        Generate all report formats.

        Args:
            comparison: Comparison results
            baseline: Optional baseline evaluation
            abliterated: Optional abliterated evaluation

        Returns:
            Dictionary mapping format to output path
        """
        results = {}

        # HTML report
        html_path = self.generate_html_report(
            comparison=comparison,
            baseline=baseline,
            abliterated=abliterated,
        )
        results["html"] = html_path

        # JSON report
        json_path = self.generate_json_report(comparison=comparison)
        results["json"] = json_path

        # CSV report
        csv_path = self.generate_csv_report(comparison=comparison)
        results["csv"] = csv_path

        logger.info(f"Generated all reports in {self.output_dir}")
        return results


def generate_all_reports(
    comparison: ComparisonResult,
    output_dir: Path | str,
    baseline: Optional[EvaluationResult] = None,
    abliterated: Optional[EvaluationResult] = None,
) -> dict[str, Path]:
    """
    Generate all report formats.

    Convenience function for quick report generation.

    Args:
        comparison: Comparison results
        output_dir: Directory for output
        baseline: Optional baseline evaluation
        abliterated: Optional abliterated evaluation

    Returns:
        Dictionary mapping format to output path
    """
    generator = ReportGenerator(output_dir=output_dir)
    return generator.generate_all(
        comparison=comparison,
        baseline=baseline,
        abliterated=abliterated,
    )
