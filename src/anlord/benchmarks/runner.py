# SPDX-License-Identifier: AGPL-3.0-or-later
"""Integration with EleutherAI's lm-evaluation-harness."""

from __future__ import annotations

import json
import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..utils.runtime import configure_huggingface_environment, prepare_optional_ml_dependencies
from .tasks import expand_task_list, get_task_config

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    task_id: str
    task_name: str
    model_id: str
    primary_metric: float = 0.0
    primary_metric_stderr: float = 0.0
    all_metrics: dict = field(default_factory=dict)
    num_samples: int = 0
    num_fewshot: int = 0
    execution_time_seconds: float = 0.0
    config: dict = field(default_factory=dict)
    hardware: dict = field(default_factory=dict)
    error: Optional[str] = None
    warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "task_name": self.task_name,
            "model_id": self.model_id,
            "primary_metric": self.primary_metric,
            "primary_metric_stderr": self.primary_metric_stderr,
            "all_metrics": _json_safe(self.all_metrics),
            "num_samples": self.num_samples,
            "num_fewshot": self.num_fewshot,
            "execution_time_seconds": self.execution_time_seconds,
            "config": _json_safe(self.config),
            "hardware": _json_safe(self.hardware),
            "error": self.error,
            "warnings": self.warnings,
        }

    @property
    def formatted_score(self) -> str:
        return f"{self.primary_metric:.4f}"

    @property
    def succeeded(self) -> bool:
        return self.error is None


class BenchmarkRunner:
    """Run standardized benchmark tasks against one Hugging Face model."""

    def __init__(
        self,
        model_id: str,
        output_dir: Path | str,
        dtype: str = "auto",
        device: str = "cuda",
        device_map: str = "auto",
        batch_size: int = 1,
        seed: int = 42,
        revision: str | None = None,
        quantization: str = "none",
        cache_dir: Path | str | None = None,
        max_memory: dict | None = None,
    ):
        self.model_id = model_id
        self.output_dir = Path(output_dir)
        self.dtype = dtype
        self.device = device
        self.device_map = device_map
        self.batch_size = batch_size
        self.seed = seed
        self.revision = revision
        self.quantization = quantization
        self.cache_dir = Path(cache_dir).resolve() if cache_dir else None
        self.max_memory = max_memory
        self._loaded_model = None
        self._loaded_tokenizer = None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.cache_dir:
            configure_huggingface_environment(self.cache_dir)

    @staticmethod
    def _normalize_max_memory(max_memory: dict | None) -> dict | None:
        if not max_memory:
            return max_memory
        normalized: dict = {}
        for key, value in max_memory.items():
            # accelerate expects int keys for GPUs, but JSON/planner uses "0" string
            try:
                # if key is "0", "1", etc. convert to int
                if isinstance(key, str) and key.isdigit():
                    normalized[int(key)] = value
                else:
                    normalized[key] = value
            except Exception:
                normalized[key] = value
        return normalized

    def _get_model_args(self) -> dict[str, Any]:
        """Build HFLM constructor arguments.

        ``batch_size`` and ``device`` intentionally do not appear here.  The
        harness passes them as additional constructor arguments, so including
        either in ``model_args`` causes ``got multiple values`` errors.

        Do not pass a live model object or ``quantization_config`` here.
        HFLM 0.4.x already forwards ``quantization_config`` into
        ``_create_model``, and a live ``nn.Module`` makes ``dump_config``
        deepcopy GPU weights until an 8 GB card OOMs.
        """
        args: dict[str, Any] = {
            "pretrained": self.model_id,
            "dtype": self.dtype,
            "trust_remote_code": True,
        }
        if self.revision:
            args["revision"] = self.revision
        if self.cache_dir:
            args["cache_dir"] = str(self.cache_dir / "hub")
        # Do not pass device_map here for quantized loads. HFLM 0.4.x sees
        # device_map, sets "Model parallel was set to False", then forwards
        # device_map=None into from_pretrained — the 9B model lands on CPU.
        if self.quantization not in {"bnb_4bit", "bnb_8bit"}:
            if self.device_map and self.device_map != "cpu":
                args["device_map"] = self.device_map
            if self.max_memory:
                args["max_memory"] = self._normalize_max_memory(self.max_memory)
        return args

    def _quantization_config_kwargs(self, *, four_bit: bool) -> dict[str, Any]:
        compute = self.dtype if self.dtype not in {"auto", "float32"} else "bfloat16"
        if four_bit:
            return {
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_compute_dtype": compute,
                "llm_int8_enable_fp32_cpu_offload": True,
            }
        return {
            "load_in_8bit": True,
            "llm_int8_enable_fp32_cpu_offload": True,
        }

    @staticmethod
    def _empty_cuda_cache() -> None:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            return

    @contextmanager
    def _lm_eval_runtime_patches(self):
        """Inject 4-bit loading without handing GPU modules to lm-eval."""
        dump_original = None
        from_pretrained_original = None
        auto_model = None
        try:
            try:
                from lm_eval.api.task import Task

                dump_original = Task.dump_config

                def dump_config(task_self):
                    try:
                        return dump_original(task_self)
                    except Exception as error:
                        message = str(error).lower()
                        if (
                            "out of memory" in message
                            or type(error).__name__ == "OutOfMemoryError"
                        ):
                            logger.warning(
                                "lm-eval dump_config hit GPU OOM; omitting task config snapshot"
                            )
                            self._empty_cuda_cache()
                            return {}
                        raise

                Task.dump_config = dump_config
            except Exception:
                dump_original = None

            if self.quantization in {"bnb_4bit", "bnb_8bit"}:
                from transformers import AutoModelForCausalLM, BitsAndBytesConfig

                auto_model = AutoModelForCausalLM
                from_pretrained_original = AutoModelForCausalLM.from_pretrained
                quant = BitsAndBytesConfig(
                    **self._quantization_config_kwargs(
                        four_bit=self.quantization == "bnb_4bit"
                    )
                )
                device_map = (
                    self.device_map if self.device_map and self.device_map != "cpu" else None
                )
                max_memory = self.max_memory

                def from_pretrained(*args, **kwargs):
                    kwargs.pop("load_in_4bit", None)
                    kwargs.pop("load_in_8bit", None)
                    kwargs["quantization_config"] = quant
                    if device_map:
                        kwargs["device_map"] = device_map
                    if max_memory:
                        kwargs["max_memory"] = self._normalize_max_memory(max_memory)
                    logger.info("Loading 4-bit weights onto GPU for lm-eval")
                    model = from_pretrained_original(*args, **kwargs)
                    logger.info("GPU 4-bit load finished; starting benchmark")
                    return model

                AutoModelForCausalLM.from_pretrained = staticmethod(from_pretrained)
            yield
        finally:
            if dump_original is not None:
                from lm_eval.api.task import Task

                Task.dump_config = dump_original
            if auto_model is not None and from_pretrained_original is not None:
                auto_model.from_pretrained = from_pretrained_original
            self._empty_cuda_cache()

    def _ensure_model(self) -> tuple[Any, Any]:
        """Load the 4-bit model once and reuse it for every benchmark."""
        if self._loaded_model is not None and self._loaded_tokenizer is not None:
            return self._loaded_model, self._loaded_tokenizer

        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        hub = str(self.cache_dir / "hub") if self.cache_dir else None
        shared: dict[str, Any] = {"trust_remote_code": True}
        if self.revision:
            shared["revision"] = self.revision
        if hub:
            shared["cache_dir"] = hub

        model_kwargs = dict(shared)
        model_kwargs["dtype"] = self.dtype
        if self.quantization in {"bnb_4bit", "bnb_8bit"}:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                **self._quantization_config_kwargs(four_bit=self.quantization == "bnb_4bit")
            )
        if self.device_map and self.device_map != "cpu":
            model_kwargs["device_map"] = self.device_map
        elif self.device == "cuda":
            model_kwargs["device_map"] = "cuda"
        if self.max_memory:
            model_kwargs["max_memory"] = self._normalize_max_memory(self.max_memory)

        logger.info("Loading Hugging Face model once for all remaining benchmarks: %s", self.model_id)
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, **shared)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(self.model_id, **model_kwargs)
        model.eval()
        self._loaded_model = model
        self._loaded_tokenizer = tokenizer
        logger.info("Cached GPU model for remaining lm-eval tasks")
        return model, tokenizer

    def _release_model(self) -> None:
        self._loaded_model = None
        self._loaded_tokenizer = None
        self._empty_cuda_cache()

    def run_benchmark(
        self,
        task_id: str,
        num_fewshot: int | None = None,
        limit: int | None = None,
        apply_chat_template: bool = False,
        system_instruction: str | None = None,
    ) -> BenchmarkResult:
        task_config = get_task_config(task_id)
        task_name = task_config.name if task_config else task_id
        harness_task_id = task_config.task_id if task_config else task_id
        result = BenchmarkResult(task_id=task_id, task_name=task_name, model_id=self.model_id)

        if num_fewshot is None and task_config:
            num_fewshot = task_config.num_fewshot
        result.num_fewshot = num_fewshot or 0

        logger.info("Running benchmark: %s on %s", task_name, self.model_id)
        start_time = time.time()
        try:
            prepare_optional_ml_dependencies()
            from lm_eval import simple_evaluate

            model, tokenizer = self._ensure_model()
            model_args = {
                "pretrained": model,
                "tokenizer": tokenizer,
                "trust_remote_code": True,
            }
            gen_kwargs = (
                task_config.gen_kwargs.copy() if task_config and task_config.gen_kwargs else None
            )
            logger.info(
                "Starting lm-eval for %s (silent gap after weight load is 4-bit/CUDA setup)",
                task_name,
            )
            stop_heartbeat = threading.Event()

            def heartbeat() -> None:
                started = time.time()
                while not stop_heartbeat.wait(30):
                    logger.info(
                        "lm-eval %s still running (%.0f s elapsed)",
                        task_name,
                        time.time() - started,
                    )

            pulse = threading.Thread(target=heartbeat, daemon=True)
            pulse.start()
            try:
                with self._lm_eval_runtime_patches():
                    results = simple_evaluate(
                        model="hf",
                        model_args=model_args,
                        tasks=[harness_task_id],
                        num_fewshot=result.num_fewshot,
                        batch_size=self.batch_size,
                        device=self.device,
                        limit=limit,
                        log_samples=False,
                        gen_kwargs=gen_kwargs,
                        apply_chat_template=apply_chat_template,
                        system_instruction=system_instruction,
                        random_seed=self.seed,
                        numpy_random_seed=self.seed,
                        torch_random_seed=self.seed,
                        fewshot_random_seed=self.seed,
                    )
            finally:
                stop_heartbeat.set()
            if not results:
                raise RuntimeError("lm-eval returned no results")

            task_results = self._find_task_results(results, harness_task_id)
            if not task_results:
                available = sorted(set(results.get("results", {})) | set(results.get("groups", {})))
                raise RuntimeError(
                    f"lm-eval returned no metrics for {harness_task_id!r}; "
                    f"available entries: {available}"
                )

            primary = task_config.primary_metric if task_config else "acc"
            score_key = self._find_metric_key(task_results, primary)
            if score_key is None:
                numeric_keys = [
                    key
                    for key, value in task_results.items()
                    if key != "alias" and "stderr" not in key and isinstance(value, (int, float))
                ]
                if not numeric_keys:
                    raise RuntimeError(
                        f"No numeric metrics found for {harness_task_id}: {sorted(task_results)}"
                    )
                score_key = numeric_keys[0]
                result.warnings.append(
                    f"Primary metric {primary!r} was unavailable; used {score_key!r}."
                )

            score = task_results[score_key]
            result.primary_metric = float(score)
            stderr_key = self._find_stderr_key(task_results, score_key)
            if stderr_key:
                result.primary_metric_stderr = float(task_results[stderr_key])
            result.all_metrics = _json_safe(task_results)
            result.num_samples = self._get_sample_count(results, harness_task_id)
            result.config = {
                "task_id": task_id,
                "harness_task_id": harness_task_id,
                "primary_metric": score_key,
                "num_fewshot": result.num_fewshot,
                "limit": limit,
                "model_args": _json_safe(self._get_model_args()),
                "batch_size": self.batch_size,
                "device": self.device,
                "seed": self.seed,
                "cache_dir": str(self.cache_dir) if self.cache_dir else None,
            }
            logger.info(
                "Completed %s: %.4f (%.1fs)",
                task_name,
                result.primary_metric,
                time.time() - start_time,
            )
        except ImportError as error:
            result.error = f"lm-eval is not installed: {error}"
            logger.error("%s", result.error)
        except Exception as error:
            result.error = str(error)
            logger.error("Benchmark %s failed: %s", task_name, error)
            logger.debug("Benchmark traceback", exc_info=True)
        finally:
            result.execution_time_seconds = time.time() - start_time
        return result

    @staticmethod
    def _find_task_results(results: dict, task_id: str) -> dict:
        return results.get("results", {}).get(task_id) or results.get("groups", {}).get(task_id, {})

    @staticmethod
    def _find_metric_key(metrics: dict, primary: str) -> str | None:
        if primary in metrics:
            return primary
        preferred = f"{primary},none"
        if preferred in metrics:
            return preferred
        for key in metrics:
            if key.split(",", 1)[0] == primary and "stderr" not in key:
                return key
        return None

    @staticmethod
    def _find_stderr_key(metrics: dict, score_key: str) -> str | None:
        metric, separator, filter_name = score_key.partition(",")
        candidates = [
            f"{metric}_stderr,{filter_name}" if separator else f"{metric}_stderr",
            f"{metric},stderr",
        ]
        return next((key for key in candidates if key in metrics), None)

    @staticmethod
    def _get_sample_count(results: dict, task_id: str) -> int:
        sample_data = results.get("n-samples", {})

        def effective_count(value: Any) -> int:
            if isinstance(value, dict):
                return int(value.get("effective", value.get("original", 0)))
            return int(value or 0)

        if task_id in sample_data:
            return effective_count(sample_data[task_id])
        subtasks = results.get("group_subtasks", {}).get(task_id, [])
        return sum(effective_count(sample_data.get(task, 0)) for task in subtasks)

    def run_multiple_benchmarks(
        self,
        task_ids: list[str],
        num_fewshot: int | None = None,
        limit: int | None = None,
        skip_existing: bool = True,
        save_results: bool = True,
    ) -> list[BenchmarkResult]:
        expanded_tasks = expand_task_list(task_ids)
        results: list[BenchmarkResult] = []
        fatal_error: str | None = None
        logger.info("Running %s benchmarks...", len(expanded_tasks))
        try:
            for task_id in expanded_tasks:
                task_config = get_task_config(task_id)
                task_name = task_config.name if task_config else task_id
                if fatal_error:
                    result = BenchmarkResult(
                        task_id=task_id,
                        task_name=task_name,
                        model_id=self.model_id,
                        error=f"Skipped after model initialization failed: {fatal_error}",
                    )
                    if save_results:
                        self._save_result(result)
                    results.append(result)
                    continue

                existing = self.load_result(self.output_dir / f"{task_id}.json")
                if (
                    skip_existing
                    and existing is not None
                    and existing.succeeded
                    and existing.model_id == self.model_id
                    and existing.config.get("num_fewshot") == (num_fewshot or 0)
                    and existing.config.get("limit") == limit
                    and existing.config.get("batch_size") == self.batch_size
                    and existing.config.get("device") == self.device
                    and existing.config.get("seed") == self.seed
                    and existing.config.get("cache_dir")
                    == (str(self.cache_dir) if self.cache_dir else None)
                ):
                    logger.info("Skipping completed benchmark: %s", task_id)
                    result = existing
                else:
                    result = self.run_benchmark(task_id, num_fewshot=num_fewshot, limit=limit)
                    if save_results:
                        self._save_result(result)
                results.append(result)
                if result.succeeded:
                    self._log_progress(len(results), len(expanded_tasks), result)
                elif self._is_model_initialization_error(result.error):
                    fatal_error = result.error
                    logger.error(
                        "Stopping benchmark retries because model initialization failed; "
                        "remaining tasks will be marked as skipped"
                    )
        finally:
            self._release_model()
        return results

    @staticmethod
    def _is_model_initialization_error(error: str | None) -> bool:
        if not error:
            return False
        message = error.lower()
        if "dump_config" in message:
            return False
        markers = (
            "could not load this library",
            "dll load failed",
            "cannot load library",
            "failed to import",
            "error loading model",
            "no module named",
            "os error 1455",
            "paging file",
            "файл подкачки",
            "dispatched on the cpu or the disk",
            "unexpected keyword argument",
            "load_in_4bit",
            "load_in_8bit",
            "multiple values for keyword argument",
        )
        return any(marker in message for marker in markers)

    def _save_result(self, result: BenchmarkResult) -> None:
        filepath = self.output_dir / f"{result.task_id}.json"
        temporary = filepath.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
        temporary.replace(filepath)
        logger.debug("Saved result to %s", filepath)

    @staticmethod
    def _log_progress(current: int, total: int, result: BenchmarkResult) -> None:
        logger.info(
            "[%s/%s] %s: %.4f (%.1fs)",
            current,
            total,
            result.task_name,
            result.primary_metric,
            result.execution_time_seconds,
        )

    @staticmethod
    def load_result(filepath: Path | str) -> BenchmarkResult | None:
        filepath = Path(filepath)
        if not filepath.is_file():
            return None
        try:
            data = json.loads(filepath.read_text(encoding="utf-8"))
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
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            logger.warning("Could not load benchmark result %s: %s", filepath, error)
            return None

    @classmethod
    def load_results(cls, results_dir: Path | str) -> dict[str, BenchmarkResult]:
        results: dict[str, BenchmarkResult] = {}
        results_dir = Path(results_dir)
        if not results_dir.exists():
            return results
        for filepath in results_dir.glob("*.json"):
            if filepath.name == "evaluation.json":
                continue
            result = cls.load_result(filepath)
            if result is not None:
                results[result.task_id] = result
        return results


def _json_safe(value: Any) -> Any:
    """Convert NumPy-like scalar/container values to JSON-compatible values."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value
