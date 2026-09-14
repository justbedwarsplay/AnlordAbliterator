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

from ..heretic.wrapper import HereticResult, HereticWrapper

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    model_id: str
    evaluation_type: str
    heretic: Optional[HereticResult] = None
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
            "heretic": self.heretic.to_dict() if self.heretic else None,
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
        if self.error or self.heretic is None or self.heretic.error:
            return False
        return all(
            task_id in self.benchmarks and self.benchmarks[task_id].succeeded
            for task_id in benchmarks
        )

    def heretic_succeeded(self) -> bool:
        return self.heretic is not None and not self.heretic.error


class BaselineEvaluator:
    """Evaluate the original model before abliteration."""

    def __init__(
        self,
        model_id: str,
        output_dir: Path | str,
        heretic_wrapper: Optional[HereticWrapper] = None,
        heretic_output_dir: Path | str | None = None,
        dtype: str = "auto",
        device_map: str = "auto",
        device: str = "cuda",
        batch_size: int = 1,
        seed: int = 42,
        quantization: str = "none",
        model_commit: str | None = None,
        cache_dir: Path | str | None = None,
        max_memory: dict | None = None,
        heretic_batch_size: int | None = None,
        heretic_max_batch_size: int | None = None,
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

        if heretic_wrapper is None:
            self.heretic = HereticWrapper(
                model_id=model_id,
                output_dir=heretic_output_dir or self.output_dir,
                dtype=dtype,
                device=device,
                device_map=device_map,
                quantization=quantization,
                seed=seed,
                model_commit=model_commit,
                cache_dir=self.cache_dir,
                max_memory=max_memory,
                heretic_batch_size=heretic_batch_size,
                heretic_max_batch_size=heretic_max_batch_size,
                **kwargs,
            )
        else:
            self.heretic = heretic_wrapper

    def run_evaluation(
        self,
        benchmarks: list[str],
        num_fewshot: int = 0,
        limit: Optional[int] = None,
        reuse_heretic: Optional[HereticResult] = None,
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

        if reuse_heretic is not None and not reuse_heretic.error:
            logger.info("Reusing completed Heretic baseline evaluation")
            result.heretic = reuse_heretic
        else:
            logger.info("Running Heretic evaluation on original model...")
            heretic_metrics = self.heretic.run_evaluation(model_path=None, save_results=True)
            if heretic_metrics.get("error"):
                errors.append(f"Heretic evaluation: {heretic_metrics['error']}")
            else:
                total_prompts = int(heretic_metrics.get("total_prompts", 0))
                initial_refusals = int(heretic_metrics.get("initial_refusals", 0))
                final_refusals = int(heretic_metrics.get("final_refusals", initial_refusals))
                result.heretic = HereticResult(
                    model_id=self.model_id,
                    initial_refusals=initial_refusals,
                    final_refusals=final_refusals,
                    total_prompts=total_prompts,
                    initial_refusal_rate=(
                        initial_refusals / total_prompts if total_prompts else 0.0
                    ),
                    final_refusal_rate=(final_refusals / total_prompts if total_prompts else 0.0),
                    kl_divergence=float(heretic_metrics.get("kl_divergence", 0.0)),
                    evaluation_time_seconds=float(heretic_metrics.get("evaluation_time", 0.0)),
                )

        if errors:
            result.duration_seconds = time.time() - start_time
            result.error = "; ".join(errors)
            self._save_results(result)
            logger.error("Baseline refusal evaluation failed; benchmarks will not be started")
            return result

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
                skip_existing=reuse_heretic is not None,
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
            heretic_data = data.get("heretic")
            heretic_result = _load_heretic_result(heretic_data) if heretic_data else None
            benchmarks = {
                task_id: _load_benchmark_result(benchmark_data)
                for task_id, benchmark_data in data.get("benchmarks", {}).items()
            }
            return EvaluationResult(
                model_id=data["model_id"],
                evaluation_type=data["evaluation_type"],
                heretic=heretic_result,
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


def _load_heretic_result(data: dict) -> HereticResult:
    model_id = data.get("model") or data.get("model_id")
    if not model_id:
        raise KeyError("heretic result is missing model id")
    return HereticResult(
        model_id=model_id,
        abliterated_model_path=data.get("abliterated_model"),
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
    if baseline.heretic and abliterated.heretic:
        comparison["refusal_change"] = {
            "initial_refusals": {
                "baseline": baseline.heretic.initial_refusals,
                "abliterated": abliterated.heretic.initial_refusals,
                "delta": (abliterated.heretic.initial_refusals - baseline.heretic.initial_refusals),
            },
            "final_refusals": {
                "baseline": baseline.heretic.final_refusals,
                "abliterated": abliterated.heretic.final_refusals,
                "delta": (abliterated.heretic.final_refusals - baseline.heretic.final_refusals),
            },
            "kl_divergence": abliterated.heretic.kl_divergence,
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
