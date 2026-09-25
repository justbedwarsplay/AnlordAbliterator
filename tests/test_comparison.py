# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Tests for Anlord Abliterator comparison module.
"""

import pytest

from anlord.evaluation.comparison import (
    ComparisonResult,
    compare_results,
    generate_delta_table,
    format_comparison_for_markdown,
)


class TestComparisonResult:
    """Test ComparisonResult class."""

    def test_default_result(self):
        """Test creating default comparison result."""
        result = ComparisonResult(model_id="test/model")

        assert result.model_id == "test/model"
        # Unmeasured metrics default to None ("n/a"), not fake zeros.
        assert result.kl_divergence is None
        assert result.final_refusals_baseline is None
        assert result.refusal_reduction is None
        assert result.benchmarks == {}
        assert len(result.warnings) == 0

    def test_refusal_reduction(self):
        """Test refusal reduction calculation."""
        result = ComparisonResult(
            model_id="test/model",
            final_refusals_baseline=100,
            final_refusals_abliterated=20,
        )

        assert result.refusal_reduction == 80
        assert result.refusal_reduction_percent == 80.0

    def test_refusal_reduction_zero_baseline(self):
        """Test refusal reduction with zero baseline."""
        result = ComparisonResult(
            model_id="test/model",
            final_refusals_baseline=0,
            final_refusals_abliterated=0,
        )

        assert result.refusal_reduction == 0
        assert result.refusal_reduction_percent == 0.0

    def test_to_dict(self):
        """Test converting to dictionary."""
        result = ComparisonResult(
            model_id="test/model",
            kl_divergence=0.05,
            final_refusals_baseline=100,
            final_refusals_abliterated=10,
        )

        data = result.to_dict()

        assert data["model_id"] == "test/model"
        assert data["kl_divergence"] == 0.05
        assert data["final_refusals_baseline"] == 100


class TestCompareResults:
    """Test compare_results function."""

    def test_empty_comparison(self):
        """Test comparison with empty results."""
        from anlord.evaluation.evaluator import EvaluationResult

        baseline = EvaluationResult(
            model_id="test/model",
            evaluation_type="baseline",
        )
        abliterated = EvaluationResult(
            model_id="test/model",
            evaluation_type="abliterated",
        )

        result = compare_results(baseline, abliterated)

        assert result.model_id == "test/model"
        # No abliteration metrics -> unknown, not fake zeros.
        assert result.final_refusals_baseline is None


class TestDeltaTable:
    """Test delta table generation."""

    def test_generate_delta_table(self):
        """Test generating delta table."""
        result = ComparisonResult(
            model_id="test/model",
            final_refusals_baseline=100,
            final_refusals_abliterated=10,
            kl_divergence=0.05,
            benchmarks={
                "mmlu": {
                    "name": "MMLU",
                    "baseline": 0.8,
                    "abliterated": 0.79,
                    "absolute_delta": -0.01,
                    "relative_delta_percent": -1.25,
                    "baseline_stderr": 0.01,
                    "abliterated_stderr": 0.01,
                    "baseline_samples": 100,
                    "abliterated_samples": 100,
                }
            },
        )

        table = generate_delta_table(result)

        assert "MMLU" in table
        assert "0.8000" in table
        assert "0.7900" in table


class TestMarkdown:
    """Test Markdown formatting."""

    def test_format_markdown(self):
        """Test formatting comparison as Markdown."""
        result = ComparisonResult(
            model_id="test/model",
            kl_divergence=0.05,
            final_refusals_baseline=100,
            final_refusals_abliterated=10,
        )

        md = format_comparison_for_markdown(result)

        assert "# Abliteration Results for test/model" in md
        assert "KL divergence is NOT a percentage" in md


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
