# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Tests for Anlord Abliterator configuration module.
"""

import pytest
from pathlib import Path

from anlord.config import (
    Settings,
    RunMode,
    BenchmarkConfig,
    get_benchmark_configs,
    AVAILABLE_BENCHMARKS,
)


class TestSettings:
    """Test Settings class."""

    def test_default_settings(self):
        """Test default settings are correct."""
        settings = Settings()

        assert settings.model == "unsloth/gpt-oss-20b-BF16"
        assert settings.output_dir == Path("./output")
        assert settings.heretic_trials == 100
        assert settings.heretic_timeout == 43200
        assert settings.mode == RunMode.QUICK
        assert settings.seed == 42
        assert settings.quantization == "auto"
        assert settings.device_map == "auto"

    def test_custom_settings(self):
        """Test custom settings."""
        settings = Settings(
            model="Qwen/Qwen2.5-7B",
            heretic_trials=200,
            mode=RunMode.FULL,
        )

        assert settings.model == "Qwen/Qwen2.5-7B"
        assert settings.heretic_trials == 200
        assert settings.mode == RunMode.FULL

    def test_benchmarks_parsing(self):
        """Test benchmark list parsing."""
        settings = Settings(benchmarks="mmlu,gsm8k,hellaswag")

        assert settings.benchmarks == ["mmlu", "gsm8k", "hellaswag"]

    def test_get_results_dir(self):
        """Test results directory path."""
        settings = Settings(output_dir=Path("./output"))

        assert settings.get_results_dir() == Path("./output/results")
        assert settings.get_baseline_results_dir() == Path("./output/results/baseline")
        assert settings.get_abliterated_results_dir() == Path("./output/results/abliterated")
        assert settings.get_reports_dir() == Path("./output/reports")


class TestRunMode:
    """Test RunMode enum."""

    def test_run_mode_values(self):
        """Test RunMode values."""
        assert RunMode.QUICK.value == "quick"
        assert RunMode.FULL.value == "full"


class TestBenchmarkConfigs:
    """Test benchmark configuration functions."""

    def test_get_benchmark_configs(self):
        """Test getting benchmark configurations."""
        configs = get_benchmark_configs(["mmlu", "gsm8k"])

        assert len(configs) == 2
        assert configs[0].name == "MMLU"
        assert configs[1].name == "GSM8K"

    def test_available_benchmarks(self):
        """Test that all expected benchmarks are available."""
        expected = ["mmlu", "gsm8k", "hellaswag", "arc_challenge", "winogrande", "truthfulqa"]

        for benchmark in expected:
            assert benchmark in AVAILABLE_BENCHMARKS


class TestBenchmarkConfig:
    """Test BenchmarkConfig dataclass."""

    def test_default_config(self):
        """Test default benchmark config."""
        config = BenchmarkConfig(
            name="Test",
            task_id="test",
        )

        assert config.num_fewshot == 0
        assert config.description == ""
        assert config.enabled is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
