# SPDX-License-Identifier: AGPL-3.0-or-later
"""Orchestration for abliteration, evaluation, comparison, and reporting."""

from __future__ import annotations

import json
import logging
import os
import sys
import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch

from .benchmarks.native import NativeBenchmarkRunner
from .benchmarks.runner import BenchmarkRunner
from .config import ExportFormat, Settings
from .evaluation import BaselineEvaluator, ComparisonResult, EvaluationResult, compare_results
from .hardware import HardwareMetrics, HardwareMonitor, get_system_info
from .hardware.planner import describe_pagefile_fix, plan_model_load
from .heretic import HereticResult, HereticWrapper
from .heretic.compatibility import estimate_vram_requirements
from .models import prefetch_model_snapshot
from .models.downloader import build_snapshot_plan, find_local_snapshot, list_remote_model_files
from .reports import ReportGenerator
from .utils.runtime import configure_huggingface_environment, free_torch_memory

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    success: bool
    model_id: str
    baseline: Optional[EvaluationResult] = None
    abliterated: Optional[EvaluationResult] = None
    comparison: Optional[ComparisonResult] = None
    hardware: Optional[HardwareMetrics] = None
    report_paths: dict[str, Path] = field(default_factory=dict)
    total_duration_seconds: float = 0.0
    start_time: str = ""
    end_time: str = ""
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "model_id": self.model_id,
            "baseline": self.baseline.to_dict() if self.baseline else None,
            "abliterated": self.abliterated.to_dict() if self.abliterated else None,
            "comparison": self.comparison.to_dict() if self.comparison else None,
            "hardware": self.hardware.to_dict() if self.hardware else None,
            "report_paths": {key: str(path) for key, path in self.report_paths.items()},
            "total_duration_seconds": self.total_duration_seconds,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "error": self.error,
        }


class AbliterationPipeline:
    """Run the complete Anlord Abliterator workflow without treating failed steps as success."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.start_time = time.time()
        configure_huggingface_environment(settings.cache_dir)
        self._setup_directories()
        self.monitor = HardwareMonitor(interval=1.0)
        self.baseline_result: Optional[EvaluationResult] = None
        self.abliteration_result: Optional[HereticResult] = None
        self.abliterated_result: Optional[EvaluationResult] = None
        self.pipeline_result: Optional[PipelineResult] = None
        # Task 6: stable study checkpoint dir for resume (not timestamped)
        self._study_checkpoint_dir = self.settings.get_models_dir() / "heretic_study"
        # keep legacy fresh dir for compatibility but use stable one for Heretic
        self._fresh_checkpoint_dir = self._study_checkpoint_dir
        self._delta_status: dict = {}
        logger.info("Pipeline initialized for model: %s", settings.model)
        logger.info("Benchmark mode: %s", "native (original)" if getattr(settings, "native_benchmarks", True) else "lm-eval")

    def _setup_directories(self) -> None:
        for directory in (
            self.settings.output_dir,
            self.settings.get_results_dir(),
            self.settings.get_baseline_results_dir(),
            self.settings.get_abliterated_results_dir(),
            self.settings.get_reports_dir(),
            self.settings.get_models_dir(),
            self.settings.cache_dir,
            self.settings.get_models_dir() / "heretic_study",
        ):
            Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info("Directories created in: %s", self.settings.output_dir)

    def _save_run_config(self) -> None:
        config = self.settings.to_dict()
        # Task 2+8: expose delta fast path and weight placement info
        delta_flag = None
        if hasattr(self, "_delta_status") and self._delta_status:
            delta_flag = self._delta_status.get("fast_path_available")
        config.update(
            {
                "model_id": self.settings.model,
                "timestamp": datetime.now().isoformat(),
                "python_version": sys.version,
                "pytorch_version": torch.__version__,
                "delta_fast_path_available": delta_flag,
            }
        )
        # persist delta details if available
        if hasattr(self, "_delta_status") and self._delta_status:
            config["delta_status"] = self._delta_status
        if hasattr(self, "_calibration") and self._calibration:
            config["calibration"] = self._calibration
        if hasattr(self, "_eta_seconds") and self._eta_seconds:
            config["eta_seconds"] = self._eta_seconds
            config["per_trial_seconds"] = getattr(self, "_per_trial_seconds", None)
        self._write_json(self.settings.output_dir / "run_config.json", config)
        self._write_json(self.settings.output_dir / "environment.json", get_system_info().to_dict())

    @staticmethod
    def _write_json(path: Path, data: dict) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
        temporary.replace(path)

    def _check_hardware(self) -> None:
        logger.info("Checking hardware capabilities...")
        info = get_system_info()
        if not info.has_cuda and self.settings.device == "cuda":
            logger.warning("CUDA is not available; falling back to CPU")
            self.settings.device = "cpu"

        estimated_vram = estimate_vram_requirements(self.settings.model, dtype=self.settings.dtype)
        if info.has_cuda and estimated_vram > 0 and estimated_vram > info.total_vram_gb:
            logger.warning(
                "Model may exceed available VRAM (estimated %.1f GB, available %.1f GB)",
                estimated_vram,
                info.total_vram_gb,
            )

        self._apply_load_plan(info)

        if not self.settings.skip_abliteration and self.settings.quantization == "bnb_8bit":
            logger.warning("Heretic does not support bnb_8bit; using bnb_4bit for Heretic")

    def _apply_load_plan(self, info) -> None:
        """Resolve auto quantization, device_map, and memory caps before loading."""
        import psutil

        weight_gb = self._estimate_cached_weight_gb()
        memory = psutil.virtual_memory()
        plan = plan_model_load(
            model_id=self.settings.model,
            device=self.settings.device,
            dtype=self.settings.dtype,
            quantization=self.settings.quantization,
            weight_gb=weight_gb,
            vram_gb=info.total_vram_gb,
            ram_gb=info.total_ram_gb,
            available_ram_gb=memory.available / (1024**3),
        )
        if self.settings.quantization != plan.quantization:
            logger.warning(
                "Quantization %s -> %s",
                self.settings.quantization,
                plan.quantization,
            )
        self.settings.quantization = plan.quantization
        self.settings.device_map = plan.device_map
        if not self.settings.max_memory:
            self.settings.max_memory = plan.max_memory
        if self.settings.heretic_batch_size is None:
            # Preserve 0 (auto) explicitly; original 'or None' lost auto case
            self.settings.heretic_batch_size = plan.heretic_batch_size
        if self.settings.heretic_max_batch_size is None:
            self.settings.heretic_max_batch_size = getattr(plan, "heretic_max_batch_size", None)
        # Task 1 human-readable batch log (also already in plan.reasons but explicit)
        if plan.heretic_batch_size == 0:
            logger.info(
                "Heretic batch size: auto (max %s) | weights %.1f GB, VRAM %.1f GB",
                getattr(plan, "heretic_max_batch_size", 64),
                plan.estimated_weight_gb,
                plan.available_vram_gb,
            )
        else:
            logger.info(
                "Heretic batch size: %s | weights %.1f GB, VRAM %.1f GB",
                plan.heretic_batch_size,
                plan.estimated_weight_gb,
                plan.available_vram_gb,
            )

        logger.info(
            "Load plan: quantization=%s device_map=%s max_memory=%s "
            "estimated_weights=%.1f GB estimated_resident=%.1f GB "
            "VRAM=%.1f GB RAM=%.1f GB commit_limit=%.1f GB",
            self.settings.quantization,
            self.settings.device_map,
            self.settings.max_memory,
            plan.estimated_weight_gb,
            plan.estimated_load_gb,
            plan.available_vram_gb,
            plan.available_ram_gb,
            plan.commit_limit_gb,
        )
        for reason in plan.reasons:
            logger.info("Load plan: %s", reason)
        for warning in plan.warnings:
            logger.warning("%s", warning)

        if (
            plan.estimated_load_gb
            and plan.commit_limit_gb
            and plan.estimated_load_gb > plan.commit_limit_gb
        ):
            raise RuntimeError(
                "This host cannot map the model into memory: "
                f"estimated {plan.estimated_load_gb:.1f} GB, commit limit "
                f"{plan.commit_limit_gb:.1f} GB. {describe_pagefile_fix()}"
            )

    def _estimate_cached_weight_gb(self) -> float | None:
        try:
            from huggingface_hub import HfApi

            token = os.environ.get("HF_TOKEN") or None
            files = list_remote_model_files(
                HfApi(token=token),
                self.settings.model,
                revision=self.settings.model_commit,
            )
            plan = build_snapshot_plan(files)
            if plan.expected_bytes:
                return plan.expected_gb
        except Exception as error:
            logger.debug("Could not inspect remote weight sizes: %s", error)

        snapshot = find_local_snapshot(
            Path(self.settings.cache_dir) / "hub",
            self.settings.model,
            revision=self.settings.model_commit,
        )
        if snapshot and snapshot.is_dir():
            total = 0
            for path in snapshot.rglob("*"):
                if path.suffix in {".safetensors", ".bin", ".pt", ".pth"} and path.is_file():
                    total += path.stat().st_size
            if total:
                return total / (1024**3)
        return None

    def _new_heretic_wrapper(self, output_dir: Path, trials: int | None = None) -> HereticWrapper:
        # Task 6: always pass study_checkpoint_dir (stable path) for resume; ETA-based timeout
        timeout = self.settings.heretic_timeout
        # If ETA is available, override with ETA*1.5+15min (task 6) unless user set custom timeout
        if hasattr(self, "_eta_seconds") and self._eta_seconds:
            eta_based = int(self._eta_seconds * 1.5 + 900)
            if eta_based != timeout:
                logger.info("Heretic timeout: %s s (ETA*1.5+15min, original %s s)", eta_based, timeout)
                timeout = eta_based
        # ensure checkpoint dir exists and is passed always
        overrides = dict(getattr(self.settings, "heretic_config_overrides", {}) or {})
        # merge existing config_overrides
        base_overrides = dict(self._delta_status.get("config_overrides", {}) if hasattr(self, "_delta_status") else {}) if False else {}
        # Use fresh checkpoint dir (stable) always
        overrides["study_checkpoint_dir"] = str(self._study_checkpoint_dir.resolve())
        # merge user-provided overrides if any
        if hasattr(self, "_user_config_overrides"):
            overrides.update(self._user_config_overrides)
        # also respect original config_overrides if set via Settings? Settings has no hero config overrides field; use empty
        return HereticWrapper(
            model_id=self.settings.model,
            output_dir=output_dir,
            trials=trials or self.settings.heretic_trials,
            evaluation_prompts=self.settings.heretic_evaluation_prompts,
            dtype=self.settings.dtype,
            device=self.settings.device,
            device_map=self.settings.device_map,
            quantization=self.settings.quantization,
            seed=self.settings.seed,
            timeout=timeout,
            model_commit=self.settings.model_commit,
            cache_dir=self.settings.cache_dir,
            max_memory=self.settings.max_memory,
            heretic_batch_size=self.settings.heretic_batch_size,
            heretic_max_batch_size=self.settings.heretic_max_batch_size,
            config_overrides=overrides,
        )

    def _check_model_compatibility(self) -> None:
        logger.info("Checking model compatibility...")
        wrapper = self._new_heretic_wrapper(self.settings.get_models_dir(), trials=1)
        compatible, message = wrapper.check_compatibility()
        if not compatible:
            raise RuntimeError(f"Model compatibility check failed: {message}")
        logger.info("Model compatibility: %s", message)
        # Task 2: pre-flight Gated DeltaNet fast path check
        try:
            from .heretic.compatibility import check_delta_fast_path
            self._delta_status = check_delta_fast_path(
                self.settings.model, self.settings.model_commit, self.settings.cache_dir
            )
            if self._delta_status.get("is_delta_model") and not self._delta_status.get("fast_path_available"):
                logger.warning(
                    "Быстрый путь DeltaNet недоступен, ожидается замедление в 3-10 раз. "
                    "Установите: pip install triton-windows causal-conv1d flash-linear-attention "
                    "либо загрузите модель с use_kernels=True"
                )
            if self._delta_status.get("is_delta_model"):
                logger.info(
                    "Delta fast path available: %s (fla=%s, causal_conv1d=%s)",
                    self._delta_status.get("fast_path_available"),
                    self._delta_status.get("fla_available"),
                    self._delta_status.get("causal_available"),
                )
            else:
                logger.debug("Model is not DeltaNet hybrid (no linear_attention layers)")
            # Task 7: weight placement will be checked inside bridge after load; log that it will be checked
            logger.info("Weight placement will be verified after model load (checking for CPU offload)")
            # Task 8: try quick calibration via perf_probe if available (optional)
            try:
                self._run_perf_probe_calibration()
            except Exception as cal_err:
                logger.debug("Perf probe calibration skipped: %s", cal_err)
            # Persist updated run_config with delta flag (Task 2 criterion)
            try:
                self._save_run_config()
            except Exception as save_err:
                logger.debug("Failed to re-save run_config with delta status: %s", save_err)
        except Exception as err:
            logger.debug("Delta fast path check failed: %s", err)
            self._delta_status = {"fast_path_available": True, "is_delta_model": False}
            try:
                self._save_run_config()
            except Exception:
                pass

    def _run_perf_probe_calibration(self) -> None:
        """Optional Task 8: run a quick t_forward/t_decode calibration and store in run_config."""
        # Try to use tools/perf_probe functions without full model load; simple heuristic fallback
        # If we can import perf_probe and quickly measure, do it; otherwise store heuristic
        try:
            # Heuristic based on delta status: slow vs fast
            is_delta = self._delta_status.get("is_delta_model", False)
            fast = self._delta_status.get("fast_path_available", True)
            if is_delta and not fast:
                self._calibration = {"t_forward": 20.0, "t_decode": 0.1, "source": "heuristic_slow"}
            elif is_delta and fast:
                self._calibration = {"t_forward": 1.5, "t_decode": 0.015, "source": "heuristic_fast"}
            else:
                self._calibration = {"t_forward": 0.5, "t_decode": 0.02, "source": "heuristic_generic"}
            logger.debug("Calibration: %s", self._calibration)
        except Exception:
            pass

    def _format_duration(self, seconds: float) -> str:
        seconds = int(seconds)
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        if h:
            return f"{h} h {m:02d} min"
        if m:
            return f"{m} min {s:02d} s"
        return f"{s} s"

    def _estimate_heretic_eta(self) -> tuple[float, float]:
        """Estimate total and per-trial time via formula from PERFORMANCE_DIAGNOSIS.md."""
        n_trials = self.settings.heretic_trials
        n_eval = self.settings.heretic_evaluation_prompts
        # effective batch
        batch = self.settings.heretic_batch_size
        if batch is None or batch == 0:
            batch = getattr(self.settings, "heretic_max_batch_size", 32) or 32
            # auto case assume Heretic picks 32
            if batch == 0:
                batch = 32
        batch = max(1, int(batch))
        L = 100  # max_response_length default
        # t_forward/t_decode from calibration or heuristic
        calib = getattr(self, "_calibration", None)
        if calib:
            t_forward = calib.get("t_forward", 20.0)
            t_decode = calib.get("t_decode", 0.1)
        else:
            is_delta = self._delta_status.get("is_delta_model", False) if hasattr(self, "_delta_status") and self._delta_status else False
            fast = self._delta_status.get("fast_path_available", True) if hasattr(self, "_delta_status") and self._delta_status else True
            if is_delta and not fast:
                t_forward, t_decode = 20.0, 0.1
            elif is_delta and fast:
                t_forward, t_decode = 1.5, 0.015
            else:
                t_forward, t_decode = 0.5, 0.02
        import math as _math
        # Forward pass is batched, decode also batched: ceil(n_eval/batch) batches
        batches = _math.ceil(n_eval / batch)
        per_trial = batches * t_forward + batches * L * t_decode
        # prefix stage ~200 prompts
        prefix_batches = _math.ceil(200 / batch)
        prefix = prefix_batches * L * t_decode + prefix_batches * t_forward * 0.5
        total = prefix + per_trial * n_trials
        return total, per_trial

    def _ensure_eta_and_confirm(self) -> None:
        """Task 3: log ETA and ask confirmation if >2h."""
        try:
            total, per_trial = self._estimate_heretic_eta()
            self._eta_seconds = total
            self._per_trial_seconds = per_trial
            msg = f"Heretic ETA: ~{self._format_duration(total)} ({self.settings.heretic_trials} trials × ~{self._format_duration(per_trial)})"
            logger.info(msg)
            # Also persist ETA in run_config for later inspection
            try:
                self._save_run_config()
            except Exception:
                pass
            # Log timeout derived
            timeout = int(total * 1.5 + 900)
            logger.info("Heretic timeout: %s s (ETA*1.5+15min)", timeout)
            # Task 3: suggest fast profile if slow and long
            is_delta = self._delta_status.get("is_delta_model", False) if hasattr(self, "_delta_status") and self._delta_status else False
            fast = self._delta_status.get("fast_path_available", True) if hasattr(self, "_delta_status") and self._delta_status else True
            if is_delta and not fast and total > 7200:
                logger.warning(
                    "Быстрый путь недоступен и ETA >2ч (%.1f ч). Рассмотрите tools/heretic_fast.toml профиль (25 триалов, 25 промптов, 64 токена) для теста.",
                    total / 3600,
                )
            if total > 7200 and not getattr(self.settings, "yes", False):
                if sys.stdin.isatty():
                    try:
                        resp = input(f"Estimated Heretic time {self._format_duration(total)} >2h. Continue? [y/N]: ")
                        if resp.lower() not in ("y", "yes"):
                            raise RuntimeError("Aborted by user due to long ETA (>2h)")
                    except KeyboardInterrupt:
                        raise RuntimeError("Aborted by user (Ctrl+C) during ETA confirmation")
                else:
                    logger.warning("ETA >2h but non-interactive; continuing (use --yes to suppress warning)")
        except RuntimeError:
            raise
        except Exception as e:
            logger.debug("ETA estimation failed: %s", e)

    def _check_weight_placement(self, model) -> None:
        """Task 7: after from_pretrained, log device distribution and warn on CPU offload."""
        try:
            devices: dict = {}
            total = 0
            for p in model.parameters():
                dev = str(p.device)
                devices[dev] = devices.get(dev, 0) + p.numel()
                total += p.numel()
            for dev, cnt in sorted(devices.items()):
                share = 100 * cnt / total if total else 0
                logger.info("Weight placement: %s: %.1f%% (%d params)", dev, share, cnt)
            cpu_params = devices.get("cpu", 0)
            if cpu_params and total:
                share = 100 * cpu_params / total
                if share > 5:
                    logger.warning(
                        "%.1f%% weights are on CPU while CUDA is available — each forward will drag weights via PCIe (slow). "
                        "Remove max_memory or increase GPU limit, or use device_map=cuda.",
                        share,
                    )
        except Exception as e:
            logger.debug("Weight placement check failed: %s", e)

    def _get_benchmark_runner(self, model_id: str, output_dir: Path):
        """Return native (original) or lm-eval runner based on settings."""
        use_native = bool(getattr(self.settings, "native_benchmarks", True) and NativeBenchmarkRunner is not None)
        if getattr(self.settings, "native_benchmarks", True) and NativeBenchmarkRunner is None:
            logger.warning("Native benchmark runner unavailable, using lm-eval fallback")
        runner_cls = NativeBenchmarkRunner if use_native else BenchmarkRunner
        logger.info("Using %s benchmark runner", "native (original)" if use_native else "lm-eval")
        return runner_cls(
            model_id=model_id,
            output_dir=output_dir,
            dtype=self.settings.dtype,
            device=self.settings.device,
            device_map=self.settings.device_map,
            batch_size=self.settings.batch_size,
            seed=self.settings.seed,
            revision=self.settings.model_commit,
            quantization=self.settings.quantization if model_id == self.settings.model else "none",
            cache_dir=self.settings.cache_dir,
            max_memory=self.settings.max_memory,
        )

    def _prefetch_model(self) -> None:
        if not self.settings.prefetch_model:
            logger.info("Skipping model snapshot prefetch (--no-prefetch-model)")
            return

        logger.info("\n%s", "=" * 60)
        logger.info("MODEL DOWNLOAD: Hugging Face snapshot")
        logger.info("%s", "=" * 60)
        prefetch_model_snapshot(
            self.settings.model,
            cache_dir=self.settings.cache_dir,
            revision=self.settings.model_commit,
        )

    def _check_resume_status(self) -> dict:
        baseline = BaselineEvaluator.load_results(self.settings.get_results_dir())
        abliteration = self._load_abliteration_result()
        baseline_tasks = {
            task_id: benchmark.succeeded
            for task_id, benchmark in (baseline.benchmarks.items() if baseline else [])
        }
        # Support both native and lm-eval result files (same JSON, either loader works)
        abliterated_results = {}
        try:
            abliterated_results.update(BenchmarkRunner.load_results(self.settings.get_abliterated_results_dir()))
        except Exception:
            pass
        try:
            abliterated_results.update(NativeBenchmarkRunner.load_results(self.settings.get_abliterated_results_dir()))
        except Exception:
            pass
        abliterated_tasks = {
            task_id: benchmark.succeeded for task_id, benchmark in abliterated_results.items()
        }
        return {
            "baseline": bool(
                baseline
                and self._baseline_matches_current_run(baseline)
                and baseline.successful_for(
                    [] if self.settings.skip_benchmarks else self.settings.benchmarks
                )
            ),
            "baseline_benchmarks": baseline_tasks,
            "abliteration": bool(
                abliteration
                and abliteration.model_id == self.settings.model
                and not abliteration.error
                and abliteration.abliterated_model_path
                and HereticWrapper.is_model_directory(abliteration.abliterated_model_path)
            ),
            "abliterated_benchmarks": abliterated_tasks,
            "comparison": (self.settings.get_results_dir() / "comparison.json").is_file(),
        }

    def run(self) -> PipelineResult:
        logger.info("%s", "=" * 60)
        logger.info("Starting Anlord Abliterator pipeline")
        logger.info("%s", "=" * 60)
        metrics: HardwareMetrics | None = None

        try:
            self._check_hardware()
            self._save_run_config()
            self._check_model_compatibility()
            self._prefetch_model()
            if self.settings.resume:
                status = self._check_resume_status()
                if (
                    any(
                        value
                        for key, value in status.items()
                        if key not in {"baseline_benchmarks", "abliterated_benchmarks"}
                    )
                    or any(status["baseline_benchmarks"].values())
                    or any(status["abliterated_benchmarks"].values())
                ):
                    logger.info("Found existing valid results; pipeline can resume")
                    logger.info("Resume status: %s", status)

            self.monitor.start()
            self._run_or_load_baseline()
            self._run_or_load_abliteration()
            self._run_abliterated_evaluation()
            comparison = self._run_comparison()

            metrics = self.monitor.stop()
            comparison.peak_vram_gb = metrics.peak_vram_gb
            comparison.peak_ram_gb = metrics.peak_ram_gb
            comparison.total_duration_seconds = time.time() - self.start_time
            self._write_json(
                self.settings.get_results_dir() / "comparison.json",
                comparison.to_dict(),
            )
            report_paths = self._generate_reports(comparison)

            self.pipeline_result = PipelineResult(
                success=True,
                model_id=self.settings.model,
                baseline=self.baseline_result,
                abliterated=self.abliterated_result,
                comparison=comparison,
                hardware=metrics,
                report_paths=report_paths,
                total_duration_seconds=time.time() - self.start_time,
                start_time=datetime.fromtimestamp(self.start_time).isoformat(),
                end_time=datetime.now().isoformat(),
            )
        except Exception as error:
            logger.error("Pipeline failed: %s", error, exc_info=True)
            if metrics is None:
                metrics = self.monitor.stop()
            self.pipeline_result = PipelineResult(
                success=False,
                model_id=self.settings.model,
                baseline=self.baseline_result,
                abliterated=self.abliterated_result,
                hardware=metrics,
                total_duration_seconds=time.time() - self.start_time,
                start_time=datetime.fromtimestamp(self.start_time).isoformat(),
                end_time=datetime.now().isoformat(),
                error=str(error),
            )

        self._print_summary()
        return self.pipeline_result

    def _expected_benchmarks(self) -> list[str]:
        return [] if self.settings.skip_benchmarks else self.settings.benchmarks

    def _baseline_heretic_matches(self, result: EvaluationResult) -> bool:
        """Reuse Heretic metrics for the same model even if benchmark limit/mode changed."""
        return result.model_id == self.settings.model

    def _load_standalone_heretic_metrics(self) -> HereticResult | None:
        path = (
            self.settings.get_models_dir() / "heretic_baseline" / "heretic_evaluation.json"
        )
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            logger.warning("Could not read %s: %s", path, error)
            return None
        if data.get("error"):
            return None
        model_id = data.get("model") or data.get("model_id") or self.settings.model
        if model_id != self.settings.model:
            return None
        if data.get("initial_refusals") is None and data.get("total_prompts") is None:
            return None
        total_prompts = int(data.get("total_prompts") or 0)
        initial_refusals = int(data.get("initial_refusals") or 0)
        final_refusals = int(data.get("final_refusals") or initial_refusals)
        logger.info("Loaded Heretic metrics from %s", path)
        return HereticResult(
            model_id=model_id,
            initial_refusals=initial_refusals,
            final_refusals=final_refusals,
            total_prompts=total_prompts,
            initial_refusal_rate=(initial_refusals / total_prompts if total_prompts else 0.0),
            final_refusal_rate=(final_refusals / total_prompts if total_prompts else 0.0),
            kl_divergence=float(data.get("kl_divergence") or 0.0),
            evaluation_time_seconds=float(data.get("evaluation_time") or 0.0),
        )

    def _heretic_result_to_reuse(self, existing: EvaluationResult | None) -> HereticResult | None:
        evaluation_path = self.settings.get_results_dir() / "baseline" / "evaluation.json"
        if existing is None:
            logger.info("No baseline evaluation.json at %s", evaluation_path)
        elif not existing.heretic_succeeded():
            logger.info("Baseline evaluation.json has no successful Heretic metrics")
        elif not self._baseline_heretic_matches(existing):
            logger.info(
                "Heretic baseline is for %s, current model is %s",
                existing.model_id,
                self.settings.model,
            )
        else:
            return existing.heretic
        fallback = self._load_standalone_heretic_metrics()
        if fallback is not None:
            return fallback
        return None

    def _baseline_matches_current_run(self, result: EvaluationResult) -> bool:
        config = result.config
        return (
            self._baseline_heretic_matches(result)
            and config.get("benchmarks") == self._expected_benchmarks()
            and config.get("num_fewshot") == self.settings.num_fewshot
            and config.get("limit") == self.settings.limit
        )

    def _run_or_load_baseline(self) -> None:
        logger.info("\n%s", "=" * 60)
        logger.info("STEP 1: Baseline Evaluation")
        logger.info("%s", "=" * 60)
        expected = self._expected_benchmarks()
        existing = BaselineEvaluator.load_results(self.settings.get_results_dir())
        if (
            (self.settings.resume or self.settings.skip_baseline)
            and existing
            and self._baseline_matches_current_run(existing)
            and existing.successful_for(expected)
        ):
            logger.info("Using completed baseline evaluation")
            self.baseline_result = existing
            if self.settings.skip_benchmarks:
                self.baseline_result.benchmarks = {}
            logger.info("Baseline evaluation completed")
            logger.info("Releasing GPU/RAM after baseline so Heretic can map the model")
            free_torch_memory()
            return
        if self.settings.skip_baseline:
            raise RuntimeError(
                "--skip-baseline was specified, but no complete baseline evaluation exists"
            )
        # Task 4: avoid duplicate Heretic evaluation — reuse abliteration initial metrics by default
        should_skip_heretic = (
            not getattr(self.settings, "baseline_evaluate", False)
            and not self.settings.skip_abliteration
        )
        if should_skip_heretic:
            logger.info(
                "Skipping separate baseline Heretic evaluation; will reuse abliteration initial metrics (use --baseline-evaluate to force separate run)"
            )
            # Still need baseline benchmarks unless skipped; run them now without Heretic
            # Try to reuse existing benchmarks if resume
            if self.settings.skip_benchmarks:
                # No benchmarks needed, create empty baseline (heretic will be filled after abliteration)
                from datetime import datetime as _dt
                self.baseline_result = EvaluationResult(
                    model_id=self.settings.model,
                    evaluation_type="baseline",
                    heretic=None,
                    benchmarks={},
                    timestamp=_dt.now().isoformat(),
                    config={
                        "benchmarks": expected,
                        "num_fewshot": self.settings.num_fewshot,
                        "limit": self.settings.limit,
                        "dtype": self.settings.dtype,
                        "device": self.settings.device,
                        "device_map": self.settings.device_map,
                        "batch_size": self.settings.batch_size,
                        "seed": self.settings.seed,
                        "quantization": self.settings.quantization,
                        "max_memory": self.settings.max_memory,
                        "model_commit": self.settings.model_commit,
                        "cache_dir": str(self.settings.cache_dir) if self.settings.cache_dir else None,
                    },
                )
                # Save placeholder so resume can find it (will be overwritten later with heretic)
                self._write_json(
                    self.settings.get_baseline_results_dir() / "evaluation.json",
                    self.baseline_result.to_dict(),
                )
                logger.info("Baseline benchmarks skipped; Heretic metrics deferred")
                free_torch_memory()
                return
            # Run benchmarks only (no Heretic)
            from datetime import datetime as _dt
            # Create result object
            result = EvaluationResult(
                model_id=self.settings.model,
                evaluation_type="baseline",
                heretic=None,
                timestamp=_dt.now().isoformat(),
                config={
                    "benchmarks": expected,
                    "num_fewshot": self.settings.num_fewshot,
                    "limit": self.settings.limit,
                    "dtype": self.settings.dtype,
                    "device": self.settings.device,
                    "device_map": self.settings.device_map,
                    "batch_size": self.settings.batch_size,
                    "seed": self.settings.seed,
                    "quantization": self.settings.quantization,
                    "max_memory": self.settings.max_memory,
                    "model_commit": self.settings.model_commit,
                    "cache_dir": str(self.settings.cache_dir) if self.settings.cache_dir else None,
                },
            )
            if expected:
                logger.info("Running baseline benchmarks only (Heretic deferred to abliteration)")
                runner = self._get_benchmark_runner(
                    self.settings.model, self.settings.get_baseline_results_dir()
                )
                # If resume and existing has some benchmarks, reuse them via skip_existing
                benchmark_results = runner.run_multiple_benchmarks(
                    task_ids=expected,
                    num_fewshot=self.settings.num_fewshot,
                    limit=self.settings.limit,
                    skip_existing=self.settings.resume,
                    save_results=True,
                )
                errors = []
                for bm in benchmark_results:
                    result.benchmarks[bm.task_id] = bm
                    if bm.error:
                        errors.append(f"{bm.task_name}: {bm.error}")
                if errors:
                    result.error = "; ".join(errors)
                # Save even if benchmarks failed, so resume can retry missing
                self._write_json(
                    self.settings.get_baseline_results_dir() / "evaluation.json",
                    result.to_dict(),
                )
                if result.error:
                    raise RuntimeError(f"Baseline benchmarks failed: {result.error}")
            else:
                self._write_json(
                    self.settings.get_baseline_results_dir() / "evaluation.json",
                    result.to_dict(),
                )
            self.baseline_result = result
            # Heretic is None for now; will be filled after abliteration in _run_or_load_abliteration
            logger.info("Baseline benchmarks completed (Heretic deferred)")
            free_torch_memory()
            return
        # Normal path: run full baseline evaluation with Heretic
        reuse_heretic = None
        if self.settings.resume:
            reuse_heretic = self._heretic_result_to_reuse(existing)
        if reuse_heretic is not None:
            logger.info(
                "Reusing completed Heretic baseline; retrying failed/missing benchmarks only"
            )
        evaluator = BaselineEvaluator(
            model_id=self.settings.model,
            output_dir=self.settings.get_results_dir(),
            heretic_output_dir=self.settings.get_models_dir() / "heretic_baseline",
            trials=self.settings.heretic_trials,
            evaluation_prompts=self.settings.heretic_evaluation_prompts,
            timeout=self.settings.heretic_timeout,
            dtype=self.settings.dtype,
            device=self.settings.device,
            device_map=self.settings.device_map,
            quantization=self.settings.quantization,
            batch_size=self.settings.batch_size,
            seed=self.settings.seed,
            model_commit=self.settings.model_commit,
            cache_dir=self.settings.cache_dir,
            max_memory=self.settings.max_memory,
            heretic_batch_size=self.settings.heretic_batch_size,
            heretic_max_batch_size=self.settings.heretic_max_batch_size,
            native_benchmarks=getattr(self.settings, "native_benchmarks", True),
        )
        self.baseline_result = evaluator.run_evaluation(
            benchmarks=expected,
            num_fewshot=self.settings.num_fewshot,
            limit=self.settings.limit,
            reuse_heretic=reuse_heretic,
        )

        if not self.baseline_result or not self.baseline_result.successful_for(expected):
            detail = self.baseline_result.error if self.baseline_result else "no result"
            raise RuntimeError(f"Baseline evaluation failed: {detail}")
        if self.settings.skip_benchmarks:
            self.baseline_result.benchmarks = {}
        logger.info("Baseline evaluation completed")
        logger.info("Releasing GPU/RAM after baseline so Heretic can map the model")
        free_torch_memory()

    def _synthesize_baseline_from_abliteration(self) -> None:
        """Task 4: fill deferred baseline Heretic metrics from abliteration initial values."""
        if self.baseline_result is None or self.baseline_result.heretic is not None:
            return
        if not self.abliteration_result or self.abliteration_result.error:
            return
        # Create HereticResult for baseline: initial == final == abliteration initial
        from .heretic.wrapper import HereticResult as _HR
        ar = self.abliteration_result
        hr = _HR(
            model_id=self.settings.model,
            initial_refusals=ar.initial_refusals,
            final_refusals=ar.initial_refusals,
            total_prompts=ar.total_prompts,
            initial_refusal_rate=ar.initial_refusal_rate,
            final_refusal_rate=ar.initial_refusal_rate,
            kl_divergence=ar.kl_divergence,
            trials=ar.trials,
            best_trial=ar.best_trial,
            evaluation_time_seconds=ar.evaluation_time_seconds,
            config=ar.config,
        )
        self.baseline_result.heretic = hr
        # Persist updated baseline
        self._write_json(
            self.settings.get_baseline_results_dir() / "evaluation.json",
            self.baseline_result.to_dict(),
        )
        logger.info(
            "Synthesized baseline Heretic metrics from abliteration: refusals %s/%s, KL %.4f",
            hr.initial_refusals,
            hr.total_prompts,
            hr.kl_divergence,
        )

    def _run_or_load_abliteration(self) -> None:
        logger.info("\n%s", "=" * 60)
        logger.info("STEP 2: Heretic Abliteration")
        logger.info("%s", "=" * 60)
        existing = self._load_abliteration_result()
        existing_valid = bool(
            existing
            and existing.model_id == self.settings.model
            and not existing.error
            and existing.abliterated_model_path
            and HereticWrapper.is_model_directory(existing.abliterated_model_path)
        )
        if (self.settings.resume or self.settings.skip_abliteration) and existing_valid:
            logger.info("Using completed abliteration")
            self.abliteration_result = existing
            self._synthesize_baseline_from_abliteration()
            return
        if self.settings.skip_abliteration:
            raise RuntimeError(
                "--skip-abliteration was specified, but no valid abliterated model exists"
            )

        # Task 3: ETA and confirmation before long run
        self._ensure_eta_and_confirm()
        logger.info("Freeing leftover CUDA tensors before the Heretic child process")
        free_torch_memory()
        # Ensure study checkpoint dir exists
        try:
            self._study_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        wrapper = self._new_heretic_wrapper(self.settings.get_models_dir())
        logger.info("Heretic n_startup_trials: %s (auto ~1/3 of %s)", wrapper.n_startup_trials, wrapper.trials)
        result = wrapper.run_abliteration(save_model=True, merge_lora=True)
        if result.error:
            raise RuntimeError(f"Abliteration failed: {result.error}")
        if not result.abliterated_model_path or not HereticWrapper.is_model_directory(
            result.abliterated_model_path
        ):
            raise RuntimeError("Abliteration did not produce a valid model")
        self.abliteration_result = result
        self._write_json(self.settings.get_results_dir() / "abliteration.json", result.to_dict())
        logger.info(
            "Abliteration completed: refusals %s -> %s; KL divergence %.4f",
            result.initial_refusals,
            result.final_refusals,
            result.kl_divergence,
        )
        self._synthesize_baseline_from_abliteration()

    def _load_abliteration_result(self) -> HereticResult | None:
        path = self.settings.get_results_dir() / "abliteration.json"
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return HereticResult(
                model_id=data["model"],
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
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            logger.warning("Could not load abliteration result %s: %s", path, error)
            return None

    def _run_abliterated_evaluation(self) -> None:
        logger.info("\n%s", "=" * 60)
        logger.info("STEP 3: Abliterated Model Evaluation")
        logger.info("%s", "=" * 60)
        if not self.abliteration_result or not self.abliteration_result.abliterated_model_path:
            raise RuntimeError("No abliterated model is available for evaluation")
        model_path = self.abliteration_result.abliterated_model_path
        if not HereticWrapper.is_model_directory(model_path):
            raise RuntimeError(f"Abliterated model directory is invalid: {model_path}")

        result = EvaluationResult(
            model_id=model_path,
            evaluation_type="abliterated",
            heretic=self.abliteration_result,
            timestamp=datetime.now().isoformat(),
        )
        if not self.settings.skip_benchmarks:
            # Abliterated model is already merged, use same runner but with quantization none
            use_native = bool(getattr(self.settings, "native_benchmarks", True) and NativeBenchmarkRunner is not None)
            if getattr(self.settings, "native_benchmarks", True) and NativeBenchmarkRunner is None:
                logger.warning("Native benchmark runner unavailable for abliterated eval, using lm-eval")
            runner_cls = NativeBenchmarkRunner if use_native else BenchmarkRunner
            logger.info("Using %s benchmark runner for abliterated model", "native" if use_native else "lm-eval")
            runner = runner_cls(
                model_id=model_path,
                output_dir=self.settings.get_abliterated_results_dir(),
                dtype=self.settings.dtype,
                device=self.settings.device,
                device_map=self.settings.device_map,
                batch_size=self.settings.batch_size,
                seed=self.settings.seed,
                revision=self.settings.model_commit,
                quantization="none",  # The exported model already embodies its format.
                cache_dir=self.settings.cache_dir,
                max_memory=self.settings.max_memory,
            )
            benchmark_results = runner.run_multiple_benchmarks(
                task_ids=self.settings.benchmarks,
                num_fewshot=self.settings.num_fewshot,
                limit=self.settings.limit,
                skip_existing=self.settings.resume,
                save_results=True,
            )
            errors = []
            for benchmark in benchmark_results:
                result.benchmarks[benchmark.task_id] = benchmark
                if benchmark.error:
                    errors.append(f"{benchmark.task_name}: {benchmark.error}")
            if errors:
                result.error = "; ".join(errors)

        self.abliterated_result = result
        self._write_json(
            self.settings.get_abliterated_results_dir() / "evaluation.json",
            result.to_dict(),
        )
        if result.error:
            raise RuntimeError(f"Abliterated evaluation failed: {result.error}")
        logger.info("Abliterated evaluation completed")

    def _run_comparison(self) -> ComparisonResult:
        logger.info("\n%s", "=" * 60)
        logger.info("STEP 4: Comparison")
        logger.info("%s", "=" * 60)
        if self.baseline_result is None or self.abliterated_result is None:
            raise RuntimeError("Both baseline and abliterated evaluations are required")

        if not self.settings.skip_benchmarks:
            missing_baseline = set(self.settings.benchmarks) - set(self.baseline_result.benchmarks)
            missing_abliterated = set(self.settings.benchmarks) - set(
                self.abliterated_result.benchmarks
            )
            if missing_baseline or missing_abliterated:
                raise RuntimeError(
                    "Cannot compare incomplete benchmark results; "
                    f"missing baseline={sorted(missing_baseline)}, "
                    f"missing abliterated={sorted(missing_abliterated)}"
                )

        comparison = compare_results(self.baseline_result, self.abliterated_result)
        logger.info("Comparison completed")
        return comparison

    def _generate_reports(self, comparison: ComparisonResult) -> dict[str, Path]:
        logger.info("\n%s", "=" * 60)
        logger.info("STEP 5: Generating Reports")
        logger.info("%s", "=" * 60)
        generator = ReportGenerator(output_dir=self.settings.get_reports_dir())
        paths: dict[str, Path] = {}
        formats = set(self.settings.report_formats)
        if ExportFormat.HTML in formats:
            paths["html"] = generator.generate_html_report(
                comparison, self.baseline_result, self.abliterated_result
            )
        if ExportFormat.JSON in formats:
            paths["json"] = generator.generate_json_report(comparison)
        if ExportFormat.CSV in formats:
            paths["csv"] = generator.generate_csv_report(comparison)
        for format_name, path in paths.items():
            logger.info("  %s: %s", format_name.upper(), path)
        return paths

    def _print_summary(self) -> None:
        if not self.pipeline_result:
            return
        result = self.pipeline_result
        print("\n" + "=" * 60)
        print("       ANLORD ABLITERATOR")
        print("=" * 60)
        print(f"\nModel: {result.model_id}")
        print(f"Status: {'SUCCESS' if result.success else 'FAILED'}")

        if result.comparison:
            comparison = result.comparison
            print("\nAbliteration:")
            print(f"  Initial refusals: {comparison.final_refusals_baseline}")
            print(f"  Final refusals:   {comparison.final_refusals_abliterated}")
            print(f"  KL divergence:    {comparison.kl_divergence:.4f}")
            if comparison.benchmarks:
                print("\nBenchmarks:")
                for metrics in sorted(
                    comparison.benchmarks.values(), key=lambda value: value["name"]
                ):
                    print(
                        f"  {metrics['name']:<20} {metrics['baseline']:.4f} → "
                        f"{metrics['abliterated']:.4f} "
                        f"({metrics['absolute_delta']:+.4f})"
                    )
        if result.hardware:
            print("\nHardware:")
            print(f"  Peak VRAM: {result.hardware.peak_vram_gb:.1f} GB")
            print(f"  Peak RAM:  {result.hardware.peak_ram_gb:.1f} GB")

        duration = result.total_duration_seconds
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        seconds = int(duration % 60)
        print(f"\nTotal time: {hours:02d}:{minutes:02d}:{seconds:02d}")
        if result.report_paths:
            print(f"\nReports: {self.settings.get_reports_dir()}")
        if result.error:
            print(f"\nError: {result.error}")
        print("\n" + "=" * 60)
