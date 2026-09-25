# SPDX-License-Identifier: AGPL-3.0-or-later
"""Baseline refusal and capability evaluation."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..benchmarks.runner import BenchmarkResult, BenchmarkRunner

try:
    from ..benchmarks.native import NativeBenchmarkRunner
except ImportError:
    NativeBenchmarkRunner = None  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class AbliterationResult:
    """Results from abliteration evaluation and optimization.

    Refusal metrics are None when no refusal-measuring scorer is configured
    (they are reported as "n/a" instead of fake zeros).
    """

    model_id: str
    abliterated_model_path: Optional[str] = None
    initial_refusals: Optional[int] = None
    final_refusals: Optional[int] = None
    total_prompts: Optional[int] = None
    initial_refusal_rate: Optional[float] = None
    final_refusal_rate: Optional[float] = None
    kl_divergence: Optional[float] = None
    trials: int = 0
    best_trial: int = 0
    evaluation_time_seconds: float = 0.0
    abliteration_time_seconds: float = 0.0
    total_time_seconds: float = 0.0
    config: dict = field(default_factory=dict)
    error: Optional[str] = None
    warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "model": self.model_id,
            "model_id": self.model_id,
            "abliterated_model": self.abliterated_model_path,
            "abliterated_model_path": self.abliterated_model_path,
            "initial_refusals": self.initial_refusals,
            "final_refusals": self.final_refusals,
            "total_prompts": self.total_prompts,
            "initial_refusal_rate": self.initial_refusal_rate,
            "final_refusal_rate": self.final_refusal_rate,
            "kl_divergence": self.kl_divergence,
            "trials": self.trials,
            "best_trial": self.best_trial,
            "evaluation_time_seconds": self.evaluation_time_seconds,
            "abliteration_time_seconds": self.abliteration_time_seconds,
            "total_time_seconds": self.total_time_seconds,
            "config": self.config,
            "error": self.error,
            "warnings": self.warnings,
        }

    @staticmethod
    def is_model_directory(path: Path | str) -> bool:
        """Return whether *path* looks like a loadable HF model."""
        path = Path(path)
        if not path.is_dir():
            return False
        has_config = (path / "config.json").is_file()
        has_adapter_config = (path / "adapter_config.json").is_file()
        weight_patterns = ("*.safetensors", "*.bin", "*.pt", "*.pth")
        has_weights = any(any(path.glob(pattern)) for pattern in weight_patterns)
        return has_weights and (has_config or has_adapter_config)


@dataclass
class EvaluationResult:
    model_id: str
    evaluation_type: str
    abliteration: Optional[AbliterationResult] = None
    benchmarks: dict[str, BenchmarkResult] = field(default_factory=dict)
    peak_vram_gb: float = 0.0
    peak_ram_gb: float = 0.0
    duration_seconds: float = 0.0
    timestamp: str = ""
    config: dict = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "evaluation_type": self.evaluation_type,
            "abliteration": self.abliteration.to_dict() if self.abliteration else None,
            "benchmarks": {key: value.to_dict() for key, value in self.benchmarks.items()},
            "peak_vram_gb": self.peak_vram_gb,
            "peak_ram_gb": self.peak_ram_gb,
            "duration_seconds": self.duration_seconds,
            "timestamp": self.timestamp,
            "config": self.config,
            "error": self.error,
        }

    def get_benchmark_score(self, task_id: str) -> Optional[float]:
        benchmark = self.benchmarks.get(task_id)
        return benchmark.primary_metric if benchmark else None

    def successful_for(self, benchmarks: list[str]) -> bool:
        if self.error or self.abliteration is None or self.abliteration.error:
            return False
        return all(
            task_id in self.benchmarks and self.benchmarks[task_id].succeeded
            for task_id in benchmarks
        )

    def abliteration_succeeded(self) -> bool:
        return self.abliteration is not None and not self.abliteration.error


class BaselineEvaluator:
    """Evaluate the original model before abliteration."""

    def __init__(
        self,
        model_id: str,
        output_dir: Path | str,
        abliteration_wrapper: Optional[object] = None,
        abliteration_output_dir: Path | str | None = None,
        dtype: str = "auto",
        device_map: str = "auto",
        device: str = "cuda",
        batch_size: int = 1,
        seed: int = 42,
        quantization: str = "none",
        model_commit: str | None = None,
        cache_dir: Path | str | None = None,
        max_memory: dict | None = None,
        abliteration_batch_size: int | None = None,
        abliteration_max_batch_size: int | None = None,
        native_benchmarks: bool = True,
        **kwargs,
    ):
        self.model_id = model_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir = self.output_dir / "baseline"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.dtype = dtype
        self.device = device
        self.device_map = device_map
        self.batch_size = batch_size
        self.seed = seed
        self.quantization = quantization
        self.model_commit = model_commit
        self.cache_dir = Path(cache_dir).resolve() if cache_dir else None
        self.max_memory = max_memory
        self.native_benchmarks = native_benchmarks
        # wrapper is no longer used; kept for compat

    def run_evaluation(
        self,
        benchmarks: list[str],
        num_fewshot: int = 0,
        limit: Optional[int] = None,
        reuse_abliteration: Optional[AbliterationResult] = None,
    ) -> EvaluationResult:
        start_time = time.time()
        result = EvaluationResult(
            model_id=self.model_id,
            evaluation_type="baseline",
            timestamp=datetime.now().isoformat(),
            config={
                "benchmarks": benchmarks,
                "num_fewshot": num_fewshot,
                "limit": limit,
                "dtype": self.dtype,
                "device": self.device,
                "device_map": self.device_map,
                "batch_size": self.batch_size,
                "seed": self.seed,
                "quantization": self.quantization,
                "max_memory": self.max_memory,
                "model_commit": self.model_commit,
                "cache_dir": str(self.cache_dir) if self.cache_dir else None,
            },
        )
        errors: list[str] = []
        logger.info("Starting baseline evaluation for %s", self.model_id)

        if reuse_abliteration is not None and not reuse_abliteration.error:
            logger.info("Reusing completed abliteration baseline evaluation")
            result.abliteration = reuse_abliteration
        else:
            # No separate abliteration evaluation by default; caller (pipeline) will synthesize from abliteration result.
            # If baseline_evaluate was forced, pipeline would have called this with reuse=None and expects us to run benchmarks only.
            # We create a placeholder with zero refusals; pipeline's _synthesize will fill it later.
            logger.info("Baseline abliteration metrics deferred (will be synthesized from abliteration)")
            # leave result.abliteration as None so pipeline can fill it

        if benchmarks:
            use_native = bool(self.native_benchmarks and NativeBenchmarkRunner is not None)
            if self.native_benchmarks and NativeBenchmarkRunner is None:
                logger.warning("Native benchmark runner not available, falling back to lm-eval")
            logger.info("Running benchmarks on original model... (native=%s)", use_native)
            runner_cls = NativeBenchmarkRunner if use_native else BenchmarkRunner
            runner = runner_cls(
                model_id=self.model_id,
                output_dir=self.results_dir,
                dtype=self.dtype,
                device=self.device,
                device_map=self.device_map,
                batch_size=self.batch_size,
                seed=self.seed,
                revision=self.model_commit,
                quantization=self.quantization,
                cache_dir=self.cache_dir,
                max_memory=self.max_memory,
            )
            benchmark_results = runner.run_multiple_benchmarks(
                task_ids=benchmarks,
                num_fewshot=num_fewshot,
                limit=limit,
                skip_existing=reuse_abliteration is not None,
                save_results=True,
            )
            for benchmark in benchmark_results:
                result.benchmarks[benchmark.task_id] = benchmark
                if benchmark.error:
                    errors.append(f"{benchmark.task_name}: {benchmark.error}")
                    logger.warning("  %s: ERROR - %s", benchmark.task_name, benchmark.error)
                else:
                    logger.info("  %s: %.4f", benchmark.task_name, benchmark.primary_metric)

        result.duration_seconds = time.time() - start_time
        result.error = "; ".join(errors) if errors else None
        self._save_results(result)
        logger.info("Baseline evaluation completed in %.1fs", result.duration_seconds)
        return result

    def _save_results(self, result: EvaluationResult) -> None:
        filepath = self.results_dir / "evaluation.json"
        temporary = filepath.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
        temporary.replace(filepath)
        logger.debug("Saved baseline results to %s", filepath)

    @staticmethod
    def load_results(results_dir: Path | str) -> Optional[EvaluationResult]:
        filepath = Path(results_dir) / "baseline" / "evaluation.json"
        if not filepath.exists():
            return None
        try:
            data = json.loads(filepath.read_text(encoding="utf-8"))
            abliteration_data = data.get("abliteration")
            abliteration_result = _load_abliteration_result(abliteration_data) if abliteration_data else None
            benchmarks = {
                task_id: _load_benchmark_result(benchmark_data)
                for task_id, benchmark_data in data.get("benchmarks", {}).items()
            }
            return EvaluationResult(
                model_id=data["model_id"],
                evaluation_type=data["evaluation_type"],
                abliteration=abliteration_result,
                benchmarks=benchmarks,
                peak_vram_gb=data.get("peak_vram_gb", 0.0),
                peak_ram_gb=data.get("peak_ram_gb", 0.0),
                duration_seconds=data.get("duration_seconds", 0.0),
                timestamp=data.get("timestamp", ""),
                config=data.get("config", {}),
                error=data.get("error"),
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            logger.warning("Could not load baseline evaluation %s: %s", filepath, error)
            return None


def _load_abliteration_result(data: dict) -> AbliterationResult:
    model_id = data.get("model") or data.get("model_id")
    if not model_id:
        raise KeyError("abliteration result is missing model id")
    return AbliterationResult(
        model_id=model_id,
        abliterated_model_path=data.get("abliterated_model") or data.get("abliterated_model_path"),
        initial_refusals=data.get("initial_refusals", 0),
        final_refusals=data.get("final_refusals", 0),
        total_prompts=data.get("total_prompts", 0),
        initial_refusal_rate=data.get("initial_refusal_rate", 0.0),
        final_refusal_rate=data.get("final_refusal_rate", 0.0),
        kl_divergence=data.get("kl_divergence", 0.0),
        trials=data.get("trials", 0),
        best_trial=data.get("best_trial", 0),
        evaluation_time_seconds=data.get("evaluation_time_seconds", 0.0),
        abliteration_time_seconds=data.get("abliteration_time_seconds", 0.0),
        total_time_seconds=data.get("total_time_seconds", 0.0),
        config=data.get("config", {}),
        error=data.get("error"),
        warnings=data.get("warnings", []),
    )


def _load_benchmark_result(data: dict) -> BenchmarkResult:
    return BenchmarkResult(
        task_id=data["task_id"],
        task_name=data["task_name"],
        model_id=data["model_id"],
        primary_metric=data.get("primary_metric", 0.0),
        primary_metric_stderr=data.get("primary_metric_stderr", 0.0),
        all_metrics=data.get("all_metrics", {}),
        num_samples=data.get("num_samples", 0),
        num_fewshot=data.get("num_fewshot", 0),
        execution_time_seconds=data.get("execution_time_seconds", 0.0),
        config=data.get("config", {}),
        hardware=data.get("hardware", {}),
        error=data.get("error"),
        warnings=data.get("warnings", []),
    )


def compare_models(baseline: EvaluationResult, abliterated: EvaluationResult) -> dict:
    """Return a compact dictionary comparison (legacy convenience API)."""
    comparison = {
        "model_id": baseline.model_id,
        "refusal_change": {},
        "benchmark_changes": {},
        "summary": {},
    }
    if baseline.abliteration and abliterated.abliteration:
        comparison["refusal_change"] = {
            "initial_refusals": {
                "baseline": baseline.abliteration.initial_refusals,
                "abliterated": abliterated.abliteration.initial_refusals,
                "delta": (abliterated.abliteration.initial_refusals - baseline.abliteration.initial_refusals),
            },
            "final_refusals": {
                "baseline": baseline.abliteration.final_refusals,
                "abliterated": abliterated.abliteration.final_refusals,
                "delta": (abliterated.abliteration.final_refusals - baseline.abliteration.final_refusals),
            },
            "kl_divergence": abliterated.abliteration.kl_divergence,
        }
    for task_id in set(baseline.benchmarks) & set(abliterated.benchmarks):
        baseline_score = baseline.benchmarks[task_id].primary_metric
        abliterated_score = abliterated.benchmarks[task_id].primary_metric
        delta = abliterated_score - baseline_score
        comparison["benchmark_changes"][task_id] = {
            "baseline": baseline_score,
            "abliterated": abliterated_score,
            "delta": delta,
            "relative_delta_percent": (delta / baseline_score * 100 if baseline_score else 0.0),
        }
    return comparison
