# SPDX-License-Identifier: AGPL-3.0-or-later
"""Reliable subprocess integration with Heretic."""

from __future__ import annotations

import codecs
import importlib.util
import json
import logging
import os
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Optional

from ..hardware.planner import (
    explain_model_load_failure,
    heretic_device_map_for,
    heretic_dtypes_for,
    looks_like_access_violation,
    looks_like_meta_tensor_error,
)

logger = logging.getLogger(__name__)
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def configure_cuda_allocator_env(
    environment: dict[str, str], *, is_windows: bool | None = None
) -> None:
    """Set PYTORCH_CUDA_ALLOC_CONF without using expandable_segments on Windows."""
    windows = os.name == "nt" if is_windows is None else is_windows
    allocator = environment.get("PYTORCH_CUDA_ALLOC_CONF") or os.environ.get(
        "PYTORCH_CUDA_ALLOC_CONF", ""
    )
    if windows:
        parts = [
            part
            for part in allocator.split(",")
            if part and "expandable_segments" not in part
        ]
        if parts:
            environment["PYTORCH_CUDA_ALLOC_CONF"] = ",".join(parts)
        else:
            environment.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        return
    environment["PYTORCH_CUDA_ALLOC_CONF"] = allocator or "expandable_segments:True"

_STATUS_MARKERS = (
    "loading",
    "checking",
    "detected",
    "quantized",
    "lora",
    "prompts loaded",
    "trial",
    "refusals",
    "kl divergence",
    "saving",
    "export",
    "failed",
    "error",
    "prefix",
    "running trial",
    "elapsed time",
    "estimated remaining time",
    "determining optimal batch size",
    "chosen batch size",
    "initial refusals",
    "resuming existing study",
)


# Track last significant Heretic line for heartbeat
_last_heretic_line: str = ""
_last_heretic_lock = threading.Lock()


def _log_heretic_line(line: str) -> None:
    """Show Heretic status at INFO; keep noise on DEBUG."""
    stripped = line.strip()
    if not stripped:
        return
    lowered = stripped.lower()
    is_status = stripped.startswith("*") or any(marker in lowered for marker in _STATUS_MARKERS)
    # Also capture Heretic progress markers like "Running trial i of n" without "*"
    if not is_status and ("running trial" in lowered or "batch size" in lowered):
        is_status = True
    if is_status:
        with _last_heretic_lock:
            global _last_heretic_line
            _last_heretic_line = stripped
        logger.info("%s", stripped)
        if "checking for common response prefix" in lowered:
            logger.info(
                "Heretic is generating ~200 replies to find a shared prefix. "
                "A progress bar should appear; on an 8 GB laptop this often takes "
                "15-60 minutes."
            )
        return
    logger.debug("%s", stripped)


@dataclass
class HereticResult:
    """Results from Heretic evaluation and abliteration."""

    model_id: str
    abliterated_model_path: Optional[str] = None
    initial_refusals: int = 0
    final_refusals: int = 0
    total_prompts: int = 0
    initial_refusal_rate: float = 0.0
    final_refusal_rate: float = 0.0
    kl_divergence: float = 0.0
    trials: int = 0
    best_trial: int = 0
    evaluation_time_seconds: float = 0.0
    abliteration_time_seconds: float = 0.0
    total_time_seconds: float = 0.0
    config: dict = field(default_factory=dict)
    error: Optional[str] = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "model": self.model_id,
            "abliterated_model": self.abliterated_model_path,
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

    @property
    def refusal_reduction(self) -> int:
        return self.initial_refusals - self.final_refusals

    @property
    def refusal_reduction_percent(self) -> float:
        if self.initial_refusals == 0:
            return 0.0
        return (self.refusal_reduction / self.initial_refusals) * 100


class HereticWrapper:
    """Run Heretic through its Python console entry point in an isolated process."""

    def __init__(
        self,
        model_id: str,
        output_dir: Path | str,
        trials: int = 100,
        evaluation_prompts: int = 100,
        n_startup_trials: int | None = None,
        dtype: str = "auto",
        device_map: str | None = None,
        quantization: str = "none",
        seed: int = 42,
        timeout: int = 43200,
        model_commit: str | None = None,
        cache_dir: Path | str | None = None,
        config_overrides: dict | None = None,
        max_memory: dict | None = None,
        heretic_batch_size: int | None = None,
        heretic_max_batch_size: int | None = None,
        device: str | None = None,
    ):
        self.model_id = model_id
        self.output_dir = Path(output_dir)
        self.trials = trials
        self.evaluation_prompts = evaluation_prompts
        # Task: auto n_startup_trials ~ trials/3 if not set
        if n_startup_trials is None:
            # keep Heretic default for large trials, but for n_trials < 60 we must lower it
            # ~1/3 rule from PERFORMANCE_DIAGNOSIS.md: 25->8, 100->30
            calc = max(2, trials // 3)
            # ensure at least one TPE trial remains
            if calc >= trials:
                calc = max(1, trials - 1)
            self.n_startup_trials = calc
        else:
            self.n_startup_trials = n_startup_trials
        self.dtype = dtype
        self.device_map = device_map or heretic_device_map_for(device or "cuda")
        self.quantization = "bnb_4bit" if quantization in {"bnb_4bit", "bnb_8bit"} else "none"
        self.seed = seed
        self.timeout = timeout
        self.model_commit = model_commit
        self.cache_dir = Path(cache_dir).resolve() if cache_dir else None
        self.config_overrides = config_overrides or {}
        self.max_memory = max_memory
        self.heretic_batch_size = heretic_batch_size
        # Task 1: max batch for autotune (64 when auto)
        if heretic_batch_size == 0 and heretic_max_batch_size is None:
            self.heretic_max_batch_size = 64
        else:
            self.heretic_max_batch_size = heretic_max_batch_size
        self._started_gpu_only_four_bit = (
            self.quantization == "bnb_4bit" and self.device_map == "cuda" and not self.max_memory
        )
        self._four_bit_same_retries = 0
        self._retry_pause_seconds = 5.0
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._require_heretic()

    @staticmethod
    def _require_heretic() -> None:
        """Fail early with an actionable error when Heretic is unavailable."""
        try:
            heretic_spec = importlib.util.find_spec("heretic.main")
        except (ImportError, ModuleNotFoundError):
            heretic_spec = None
        if heretic_spec is None:
            raise RuntimeError(
                "Heretic is not installed. Install it with: pip install -U heretic-llm"
            )
        try:
            installed_version = version("heretic-llm")
        except PackageNotFoundError:
            installed_version = "unknown"
        logger.info("Found Heretic %s (console entry point: heretic.main:main)", installed_version)

    def check_compatibility(self) -> tuple[bool, str]:
        from .compatibility import check_model_compatibility

        logger.info("Checking compatibility for model: %s", self.model_id)
        try:
            import transformers

            config = transformers.AutoConfig.from_pretrained(
                self.model_id,
                revision=self.model_commit,
                cache_dir=(self.cache_dir / "hub") if self.cache_dir else None,
                trust_remote_code=True,
            )
            architectures = getattr(config, "architectures", None) or [None]
            architecture = architectures[0]
            model_type = getattr(config, "model_type", "unknown")
            logger.info("Model architecture: %s", architecture)
            logger.info("Model type: %s", model_type)
            return check_model_compatibility(architecture, model_type, self.model_id)
        except Exception as error:
            logger.error("Error checking compatibility: %s", error)
            return False, f"Error loading model configuration: {error}"

    def _environment(self, evaluate_model: str | None = None) -> dict[str, str]:
        """Build version-tolerant Heretic configuration via environment variables."""
        environment = os.environ.copy()
        environment.update(
            {
                "PYTHONUTF8": "1",
                "PYTHONIOENCODING": "utf-8",
                "PYTHONUNBUFFERED": "1",
                "HERETIC_MODEL": self.model_id,
                "HERETIC_N_TRIALS": str(self.trials),
                "HERETIC_N_STARTUP_TRIALS": str(self.n_startup_trials),
                "HERETIC_SEED": str(self.seed),
                "HERETIC_DEVICE_MAP": self.device_map,
                "HERETIC_DTYPES": json.dumps(heretic_dtypes_for(self.dtype)),
                "HERETIC_QUANTIZATION": self.quantization,
                "HERETIC_EXPORT_STRATEGY": "merge",
                "HERETIC_OFFLOAD_OUTPUTS_TO_CPU": "true",
                "HERETIC_STUDY_CHECKPOINT_DIR": str((self.output_dir / "checkpoints").resolve()),
            }
        )
        configure_cuda_allocator_env(environment)

        if self.max_memory:
            environment["HERETIC_MAX_MEMORY"] = json.dumps(self.max_memory)
        # Task 1: explicit batch handling — 0 means autotune, otherwise locked
        if self.heretic_batch_size is not None:
            environment["HERETIC_BATCH_SIZE"] = str(self.heretic_batch_size)
            # Max batch caps autotune; for locked batch 1 keep same value
            if self.heretic_max_batch_size is not None:
                environment["HERETIC_MAX_BATCH_SIZE"] = str(self.heretic_max_batch_size)
            elif self.heretic_batch_size == 0:
                environment["HERETIC_MAX_BATCH_SIZE"] = "64"
            else:
                environment["HERETIC_MAX_BATCH_SIZE"] = str(self.heretic_batch_size)
        elif self.heretic_max_batch_size is not None:
            environment["HERETIC_MAX_BATCH_SIZE"] = str(self.heretic_max_batch_size)
        if self.model_commit:
            environment["HERETIC_MODEL_COMMIT"] = self.model_commit
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            environment["HF_HOME"] = str(self.cache_dir)
            environment["HF_HUB_CACHE"] = str(self.cache_dir / "hub")
            environment["HUGGINGFACE_HUB_CACHE"] = str(self.cache_dir / "hub")
            environment["HF_DATASETS_CACHE"] = str(self.cache_dir / "datasets")
        if os.name == "nt":
            environment.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        if evaluate_model:
            environment["HERETIC_EVALUATE_MODEL"] = evaluate_model

        if self.evaluation_prompts != 100:
            count = max(1, self.evaluation_prompts)
            environment["HERETIC_GOOD_EVALUATION_PROMPTS"] = json.dumps(
                {
                    "dataset": "mlabonne/harmless_alpaca",
                    "split": f"test[:{count}]",
                    "column": "text",
                }
            )
            environment["HERETIC_BAD_EVALUATION_PROMPTS"] = json.dumps(
                {
                    "dataset": "mlabonne/harmful_behaviors",
                    "split": f"test[:{count}]",
                    "column": "text",
                }
            )

        for key, value in self.config_overrides.items():
            env_key = f"HERETIC_{key.upper()}"
            environment[env_key] = json.dumps(value) if not isinstance(value, str) else value
        return environment

    def _run_bridge(
        self,
        *,
        evaluate_model: str | None = None,
        output_model_dir: Path | None = None,
    ) -> tuple[str, dict[str, Any]]:
        metrics_file = self.output_dir / (
            "heretic_evaluation_metrics.json"
            if evaluate_model
            else "heretic_abliteration_metrics.json"
        )
        metrics_file.unlink(missing_ok=True)

        # Execute the bridge by absolute path. This works both for an installed
        # ``anlord`` package and when users launch the source tree as
        # ``python -m src.anlord...`` (where the child process changes cwd).
        bridge_path = Path(__file__).with_name("bridge.py").resolve()
        command = [
            sys.executable,
            str(bridge_path),
            "--metrics-file",
            str(metrics_file.resolve()),
        ]
        if evaluate_model:
            command.append("--evaluate-only")
        else:
            if output_model_dir is None:
                raise ValueError("output_model_dir is required for abliteration")
            command.extend(["--output-dir", str(output_model_dir.resolve())])

        logger.info("Running Heretic through its Python console entry point")
        logger.debug("Heretic bridge command: %s", subprocess.list2cmdline(command))

        # Task 6: allow graceful SIGINT on Windows/Unix for checkpoint saving
        popen_kwargs: dict = {
            "cwd": self.output_dir,
            "env": self._environment(evaluate_model),
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
            "text": False,
            "bufsize": 0,
        }
        if os.name == "nt":
            popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        process = subprocess.Popen(command, **popen_kwargs)
        output_parts: list[str] = []

        def consume_output() -> None:
            """Log normal lines while rendering carriage-return progress live."""
            if process.stdout is None:
                return
            current: list[str] = []
            progress_visible = False
            decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

            def consume_character(character: str) -> None:
                nonlocal progress_visible
                output_parts.append(character)
                if character == "\r":
                    progress = "".join(current)
                    current.clear()
                    if progress.strip():
                        sys.stdout.write("\r" + progress)
                        sys.stdout.flush()
                        progress_visible = True
                elif character == "\n":
                    line = "".join(current)
                    current.clear()
                    if progress_visible:
                        if line.strip():
                            sys.stdout.write("\r" + line)
                        sys.stdout.write("\n")
                        sys.stdout.flush()
                        progress_visible = False
                    else:
                        _log_heretic_line(line)
                else:
                    current.append(character)

            while True:
                chunk = process.stdout.read(4096)
                if not chunk:
                    break
                for character in decoder.decode(chunk):
                    consume_character(character)
            for character in decoder.decode(b"", final=True):
                consume_character(character)

            trailing = "".join(current)
            if trailing:
                if progress_visible:
                    sys.stdout.write("\r" + trailing + "\n")
                    sys.stdout.flush()
                else:
                    _log_heretic_line(trailing)

        reader = threading.Thread(target=consume_output, daemon=True)
        reader.start()

        def heartbeat() -> None:
            # Task 5: show progress less often, include last Heretic line
            started = time.time()
            while process.poll() is None:
                time.sleep(300)  # every 5 minutes instead of 30s spam
                if process.poll() is None:
                    with _last_heretic_lock:
                        last = _last_heretic_line
                    suffix = f" | last: {last}" if last else ""
                    logger.info(
                        "Heretic is still running (%.0f s elapsed)%s",
                        time.time() - started,
                        suffix,
                    )

        pulse = threading.Thread(target=heartbeat, daemon=True)
        pulse.start()
        try:
            return_code = process.wait(timeout=self.timeout)
        except subprocess.TimeoutExpired as error:
            # Task 6: graceful shutdown — SIGINT then wait for checkpoint
            logger.warning(
                "Heretic timeout after %s s (%.1f min): sending SIGINT for graceful checkpoint save",
                self.timeout,
                self.timeout / 60,
            )
            try:
                if os.name == "nt":
                    process.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
                else:
                    process.send_signal(signal.SIGINT)
            except Exception as sig_error:
                logger.debug("Failed to send SIGINT: %s", sig_error)
            try:
                return_code = process.wait(timeout=90)
                logger.info("Heretic stopped gracefully after SIGINT (code %s)", return_code)
            except subprocess.TimeoutExpired:
                logger.warning("Heretic did not exit after SIGINT; killing")
                process.kill()
                process.wait()
            reader.join(timeout=5)
            if process.stdout:
                try:
                    process.stdout.close()
                except Exception:
                    pass
            raise TimeoutError(f"Heretic timed out after {self.timeout} seconds") from error
        except KeyboardInterrupt:
            logger.warning("Received interrupt; forwarding SIGINT to Heretic for checkpoint")
            try:
                if os.name == "nt":
                    process.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
                else:
                    process.send_signal(signal.SIGINT)
                process.wait(timeout=90)
            except Exception:
                process.kill()
                process.wait()
            reader.join(timeout=5)
            if process.stdout:
                try:
                    process.stdout.close()
                except Exception:
                    pass
            raise

        reader.join(timeout=5)
        if process.stdout:
            try:
                process.stdout.close()
            except Exception:
                pass

        output = "".join(output_parts).replace("\r", "\n")
        # Task 5/6: surface Resuming study info
        if "Resuming existing study" in output:
            logger.info("Resuming existing study.")
        if return_code != 0:
            raise RuntimeError(explain_model_load_failure(output, return_code))

        bridge_metrics: dict[str, Any] = {}
        if metrics_file.exists():
            try:
                bridge_metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                logger.warning("Could not read Heretic bridge metrics: %s", error)
        # Task 5: also capture elapsed/remaining if present for metrics
        elapsed_match = re.search(r"Elapsed time:\s*([0-9:]+)", output)
        remaining_match = re.search(r"Estimated remaining time:\s*([0-9:]+)", output)
        if elapsed_match:
            bridge_metrics.setdefault("heretic_elapsed", elapsed_match.group(1))
        if remaining_match:
            bridge_metrics.setdefault("heretic_remaining", remaining_match.group(1))
        return output, bridge_metrics

    @staticmethod
    def _last_output_lines(output: str, count: int = 25) -> str:
        lines = [line for line in output.splitlines() if line.strip()]
        return "\n".join(lines[-count:])

    def _should_retry_without_mixed_offload(self, error: BaseException) -> bool:
        if self.quantization != "bnb_4bit":
            return False
        text = str(error)
        return looks_like_access_violation(message=text) or looks_like_meta_tensor_error(text)

    def _pause_before_retry(self) -> None:
        if self._retry_pause_seconds > 0:
            time.sleep(self._retry_pause_seconds)

    def _keep_four_bit_on_gpu_only(self) -> None:
        """Force a pure CUDA 4-bit load with no Accelerate disk/CPU offload."""
        logger.warning("Retrying 4-bit load with device_map=cuda and no max_memory")
        self.max_memory = None
        self.device_map = "cuda"

    def _switch_to_full_precision_offload(self) -> None:
        """Fall back after a native 4-bit crash: full BF16 + CPU/pagefile offload."""
        logger.warning(
            "Retrying Heretic without quantization; leftover layers will use RAM/pagefile"
        )
        self.quantization = "none"
        self.device_map = "auto"
        if self.max_memory is None:
            self.max_memory = {"0": "6GB", "cpu": "48GB"}
            return
        updated = dict(self.max_memory)
        updated.setdefault("0", "6GB")
        current_cpu = updated.get("cpu", "0GB")
        digits = "".join(character for character in current_cpu if character.isdigit())
        cpu_gb = int(digits or "0")
        updated["cpu"] = f"{max(cpu_gb, 48)}GB"
        self.max_memory = updated

    def _apply_load_retry(self, error: BaseException) -> bool:
        """Retry a crashed 4-bit load without dumping an 18 GB BF16 model onto an 8 GB GPU."""
        if not self._should_retry_without_mixed_offload(error):
            return False

        mixed_offload = self.device_map != "cuda" or self.max_memory is not None
        if mixed_offload:
            logger.warning("4-bit mixed offload crashed; retrying GPU-only 4-bit. Cause: %s", error)
            self._keep_four_bit_on_gpu_only()
            self._pause_before_retry()
            return True

        if self._started_gpu_only_four_bit and self._four_bit_same_retries < 1:
            self._four_bit_same_retries += 1
            logger.warning(
                "4-bit GPU load crashed; waiting for VRAM to free, then retrying 4-bit. Cause: %s",
                error,
            )
            self._pause_before_retry()
            return True

        if not self._started_gpu_only_four_bit:
            logger.warning(
                "GPU-only 4-bit still failed; falling back to full precision + CPU offload. "
                "Cause: %s",
                error,
            )
            self._switch_to_full_precision_offload()
            self._pause_before_retry()
            return True

        logger.error(
            "4-bit GPU load failed twice; not falling back to BF16 because it cannot fit "
            "this GPU. Last error: %s",
            error,
        )
        return False

    def _run_bridge_with_load_retries(self, **kwargs) -> tuple[str, dict[str, Any]]:
        while True:
            try:
                return self._run_bridge(**kwargs)
            except RuntimeError as error:
                if not self._apply_load_retry(error):
                    raise

    def run_evaluation(
        self,
        model_path: str | None = None,
        save_results: bool = True,
    ) -> dict:
        target_model = model_path or self.model_id
        logger.info("Running Heretic evaluation on: %s", target_model)
        start_time = time.time()
        try:
            output, bridge_metrics = self._run_bridge_with_load_retries(
                evaluate_model=target_model
            )
            metrics = self._parse_heretic_output(output)
            metrics.update(
                {key: value for key, value in bridge_metrics.items() if value is not None}
            )
            metrics["model"] = target_model
            metrics["evaluation_time"] = time.time() - start_time
            if "initial_refusals" in metrics and "final_refusals" not in metrics:
                metrics["final_refusals"] = metrics["initial_refusals"]
            required = {"initial_refusals", "final_refusals", "total_prompts"}
            missing = required - metrics.keys()
            if missing:
                raise RuntimeError(
                    "Heretic evaluation produced no usable metrics; missing "
                    + ", ".join(sorted(missing))
                )
            if save_results:
                path = self.output_dir / "heretic_evaluation.json"
                path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            return metrics
        except Exception as error:
            logger.error("Heretic evaluation failed: %s", error)
            return {"error": str(error), "model": target_model}

    def run_abliteration(
        self,
        save_model: bool = True,
        merge_lora: bool = True,
    ) -> HereticResult:
        del merge_lora  # The bridge requests Heretic's merged export strategy.
        start_time = time.time()
        result = HereticResult(model_id=self.model_id, trials=self.trials)
        result.config = self._config_dict()

        is_compatible, message = self.check_compatibility()
        if not is_compatible:
            result.error = f"Model not compatible: {message}"
            return result

        logger.info("Starting abliteration for: %s", self.model_id)
        logger.info(
            "Heretic will load text-only CausalLM weights (vision tower skipped) "
            "so 9B 4-bit can stay on an 8 GB GPU"
        )
        output_model_dir = self.output_dir / "heretic_output" / f"abliterated-{time.time_ns()}"
        try:
            output, bridge_metrics = self._run_bridge_with_load_retries(
                output_model_dir=output_model_dir
            )
            metrics = self._parse_heretic_output(output)
            metrics.update(
                {key: value for key, value in bridge_metrics.items() if value is not None}
            )
            required = {
                "initial_refusals",
                "final_refusals",
                "total_prompts",
                "kl_divergence",
            }
            missing = required - metrics.keys()
            if missing:
                raise RuntimeError(
                    "Heretic abliteration produced no usable metrics; missing "
                    + ", ".join(sorted(missing))
                )

            result.initial_refusals = int(metrics.get("initial_refusals", 0))
            result.final_refusals = int(metrics.get("final_refusals", 0))
            result.total_prompts = int(metrics.get("total_prompts", self.evaluation_prompts))
            result.kl_divergence = float(metrics.get("kl_divergence", 0.0))
            result.best_trial = int(metrics.get("best_trial", 0))
            if result.total_prompts > 0:
                result.initial_refusal_rate = result.initial_refusals / result.total_prompts
                result.final_refusal_rate = result.final_refusals / result.total_prompts

            if save_model:
                if not self.is_model_directory(output_model_dir):
                    raise RuntimeError(
                        "Heretic returned successfully but did not export a valid model to "
                        f"{output_model_dir}"
                    )
                result.abliterated_model_path = str(output_model_dir.resolve())

            result.total_time_seconds = time.time() - start_time
            result.abliteration_time_seconds = result.total_time_seconds
            logger.info("Abliteration completed in %.1fs", result.total_time_seconds)
            logger.info("Refusals: %s -> %s", result.initial_refusals, result.final_refusals)
            logger.info("KL divergence: %s", result.kl_divergence)
        except Exception as error:
            result.error = str(error)
            result.total_time_seconds = time.time() - start_time
            logger.error("Abliteration failed: %s", error)
        return result

    @staticmethod
    def is_model_directory(path: Path | str) -> bool:
        """Return whether *path* looks like a loadable HF model or PEFT adapter."""
        path = Path(path)
        if not path.is_dir():
            return False
        has_config = (path / "config.json").is_file()
        has_adapter_config = (path / "adapter_config.json").is_file()
        weight_patterns = (
            "*.safetensors",
            "*.bin",
            "*.pt",
            "*.pth",
        )
        has_weights = any(any(path.glob(pattern)) for pattern in weight_patterns)
        return has_weights and (has_config or has_adapter_config)

    def _parse_heretic_output(self, output: str) -> dict:
        metrics: dict[str, Any] = {}
        initial_pattern = re.compile(r"Initial refusals:\s*(\d+)(?:\s*/\s*(\d+))?", re.IGNORECASE)
        refusal_pattern = re.compile(
            r"(?:^|\*\s*)Refusals:\s*(\d+)(?:\s*/\s*(\d+))?", re.IGNORECASE
        )
        kl_pattern = re.compile(
            r"KL divergence:\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)",
            re.IGNORECASE,
        )

        for raw_line in output.splitlines():
            line = _ANSI_ESCAPE.sub("", raw_line).strip()
            initial_match = initial_pattern.search(line)
            if initial_match:
                metrics["initial_refusals"] = int(initial_match.group(1))
                if initial_match.group(2):
                    metrics["total_prompts"] = int(initial_match.group(2))
                continue

            refusal_match = refusal_pattern.search(line)
            if refusal_match:
                metrics["final_refusals"] = int(refusal_match.group(1))
                if refusal_match.group(2):
                    metrics["total_prompts"] = int(refusal_match.group(2))

            kl_match = kl_pattern.search(line)
            if kl_match:
                metrics["kl_divergence"] = float(kl_match.group(1))
        return metrics

    def _config_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "model_commit": self.model_commit,
            "trials": self.trials,
            "evaluation_prompts": self.evaluation_prompts,
            "dtype": self.dtype,
            "device_map": self.device_map,
            "quantization": self.quantization,
            "max_memory": self.max_memory,
            "seed": self.seed,
            "timeout": self.timeout,
            "cache_dir": str(self.cache_dir) if self.cache_dir else None,
        }
