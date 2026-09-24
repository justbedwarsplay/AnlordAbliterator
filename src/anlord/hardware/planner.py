# SPDX-License-Identifier: AGPL-3.0-or-later
"""Choose a model load strategy that can actually fit on the host machine."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Mapping

logger = logging.getLogger(__name__)

_PARAM_COUNT = re.compile(r"(?i)(?:^|[^0-9])(\d+(?:\.\d+)?)[ _-]?b(?:illion)?(?:$|[^a-z0-9])")


@dataclass(frozen=True)
class LoadPlan:
    """Resolved Abliteration / Transformers load settings for the current host."""

    quantization: str
    device_map: str
    max_memory: dict[str, str] | None
    abliteration_dtypes: list[str]
    abliteration_batch_size: int
    abliteration_max_batch_size: int
    estimated_weight_gb: float
    estimated_load_gb: float
    available_vram_gb: float
    available_ram_gb: float
    commit_limit_gb: float
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def abliteration_dtypes_for(dtype: str) -> list[str]:
    """Return a Abliteration dtype fallback list that will not escalate to float32."""
    if dtype == "auto":
        return ["auto", "bfloat16", "float16"]
    if dtype == "bfloat16":
        return ["bfloat16", "float16"]
    if dtype == "float16":
        return ["float16", "bfloat16"]
    return [dtype]


def abliteration_device_map_for(device: str) -> str:
    """Map a user device preference to a Abliteration/Accelerate device_map."""
    return "cpu" if device == "cpu" else "auto"


def estimate_weight_gb_from_name(model_id: str) -> float | None:
    """Estimate BF16 weight size from a model id such as ``gpt-oss-20b``."""
    match = _PARAM_COUNT.search(model_id.replace("/", " "))
    if not match:
        return None
    return float(match.group(1)) * 2.0


def estimated_resident_gb(weight_gb: float, quantization: str) -> float:
    """Estimate process-resident model size after the chosen quantization."""
    if quantization == "bnb_4bit":
        return weight_gb * 0.28 + 1.5
    if quantization == "bnb_8bit":
        return weight_gb * 0.52 + 1.5
    return weight_gb * 1.15


def format_memory_budget(gigabytes: float) -> str:
    """Format a Abliteration ``max_memory`` entry."""
    rounded = max(1, int(gigabytes + 0.999))
    return f"{rounded}GB"


def get_commit_limit_gb() -> float:
    """Return RAM + pagefile/swap in GiB, or 0 when it cannot be determined."""
    if os.name == "nt":
        windows_limit = _windows_commit_limit_gb()
        if windows_limit:
            return windows_limit
    linux_limit = _linux_commit_limit_gb()
    if linux_limit:
        return linux_limit
    try:
        import psutil

        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        return (memory.total + swap.total) / (1024**3)
    except Exception:
        return 0.0


def plan_model_load(
    *,
    model_id: str,
    device: str,
    dtype: str,
    quantization: str,
    weight_gb: float | None = None,
    vram_gb: float = 0.0,
    ram_gb: float = 0.0,
    available_ram_gb: float | None = None,
    commit_limit_gb: float | None = None,
    platform_name: str | None = None,
) -> LoadPlan:
    """Pick quantization, device_map, and memory caps for this host."""
    named_weight = estimate_weight_gb_from_name(model_id) or 0.0
    estimated_weight = weight_gb if weight_gb and weight_gb > 0 else named_weight
    available = ram_gb if available_ram_gb is None else available_ram_gb
    commit_limit = commit_limit_gb if commit_limit_gb is not None else get_commit_limit_gb()
    is_windows = (platform_name or os.name) == "nt"
    reasons: list[str] = []
    warnings: list[str] = []

    requested = quantization
    resolved = requested
    if requested == "auto":
        if estimated_weight:
            full_precision = estimated_resident_gb(estimated_weight, "none")
        else:
            full_precision = 0.0
        four_bit = (
            estimated_resident_gb(estimated_weight, "bnb_4bit") if estimated_weight else 0.0
        )
        if device == "cpu":
            resolved = "bnb_4bit" if estimated_weight >= 8.0 and not is_windows else "none"
            if resolved == "none" and estimated_weight >= 8.0:
                reasons.append(
                    "Windows CPU path keeps full precision; mixed bitsandbytes offload is unstable"
                )
            else:
                reasons.append("CPU load uses 4-bit when the model is larger than 8 GB")
        elif estimated_weight and vram_gb > 0 and full_precision > vram_gb * 0.8:
            if is_windows and four_bit > vram_gb * 0.85:
                resolved = "none"
                reasons.append(
                    "4-bit weights do not fit in VRAM; on Windows mixed bitsandbytes "
                    "GPU/CPU offload often crashes with access violation 0xC0000005. "
                    "Using full precision with Accelerate CPU/pagefile offload instead"
                )
            else:
                resolved = "bnb_4bit"
                reasons.append(
                    f"Full-precision estimate {full_precision:.1f} GB exceeds 80% of "
                    f"{vram_gb:.1f} GB VRAM"
                )
        else:
            resolved = "none"
            reasons.append("Model is expected to fit in VRAM without quantization")
    elif requested == "none" and estimated_weight and vram_gb > 0:
        full_precision = estimated_resident_gb(estimated_weight, "none")
        if full_precision > vram_gb * 0.8:
            warnings.append(
                f"quantization=none keeps full precision (~{full_precision:.1f} GB). "
                f"Only {vram_gb:.1f} GB VRAM is available, so leftover layers will "
                "offload to RAM/pagefile via device_map=auto."
            )

    if resolved == "bnb_8bit":
        # Abliteration only accepts none / bnb_4bit. Keep 8-bit for the caller if they
        # asked for it, but the planner still reports a 4-bit Abliteration fallback.
        warnings.append("Abliteration does not support bnb_8bit; it will use bnb_4bit")

    abliteration_quantization = "bnb_4bit" if resolved in {"bnb_4bit", "bnb_8bit"} else "none"
    if estimated_weight:
        load_gb = estimated_resident_gb(estimated_weight, abliteration_quantization)
    else:
        load_gb = 0.0

    four_bit_fits = (
        abliteration_quantization == "bnb_4bit" and load_gb > 0 and vram_gb > 0 and load_gb <= vram_gb * 0.9
    )
    if four_bit_fits and device != "cpu":
        device_map = "cuda"
        max_memory = None
        reasons.append(
            "4-bit weights fit in VRAM; using device_map=cuda with no max_memory "
            "so Accelerate cannot offload leftover modules to disk"
        )
    else:
        device_map = abliteration_device_map_for(device)
        max_memory = _max_memory(
            device=device,
            vram_gb=vram_gb,
            ram_gb=ram_gb,
            available_ram_gb=available,
            commit_limit_gb=commit_limit,
            load_gb=load_gb,
            quantization=abliteration_quantization,
        )
        if device_map == "auto" and max_memory:
            if "cpu" in max_memory:
                reasons.append(f"CPU offload enabled via max_memory={max_memory}")
            else:
                reasons.append(f"Keeping the model on GPU via max_memory={max_memory}")

    if (
        load_gb
        and commit_limit
        and load_gb > commit_limit * 0.9
        and abliteration_quantization == "bnb_4bit"
    ):
        warnings.append(
            f"Even 4-bit loading (~{load_gb:.1f} GB) is close to the Windows/Linux "
            f"commit limit of {commit_limit:.1f} GB. Increase the pagefile/swap to "
            "at least 64 GB before retrying."
        )

    # Task 1 (rev. 3): batch choice. The native autotuner is predictive for EVERY CUDA
    # model — quantized or full precision: it measures the per-batch VRAM coefficient
    # starting from batch 1 and only attempts the next power of two when the
    # extrapolated peak still fits into the free headroom, so an oversized batch is
    # rejected on paper instead of OOM-ing the process (a bitsandbytes OOM on Windows
    # can be an uncatchable 0xC0000005). All autotune ceilings are uniform (256);
    # batch 1 is forced only for CPU/pagefile offload paths, where PCIe round-trips
    # negate batching anyway.
    if device_map == "cuda":
        gpu_allowance_gb = float("inf")
    elif max_memory and "0" in max_memory:
        try:
            gpu_allowance_gb = float(str(max_memory["0"]).removesuffix("GB"))
        except ValueError:
            gpu_allowance_gb = 0.0
    else:
        gpu_allowance_gb = 0.0
    offloads_to_ram = (
        device != "cpu"
        and bool(max_memory and "cpu" in max_memory)
        and load_gb > gpu_allowance_gb
    )

    if device == "cpu":
        # No VRAM metering available: fall back to probe-and-catch autotune, which
        # relies on OOM being a recoverable exception outside bitsandbytes kernels.
        batch_size = 0
        max_batch_size = 256
        reasons.append(
            "Abliteration batch size: probe autotune (max 256) | CPU device, trying powers of two"
        )
    elif offloads_to_ram:
        batch_size = 1
        max_batch_size = 1
        reasons.append(
            "Abliteration batch size locked to 1 to reduce peak memory "
            "(model does not fit wholly in VRAM; layers offload to CPU/pagefile)"
        )
    else:
        batch_size = 0
        max_batch_size = 256
        reasons.append(
            f"Abliteration batch size: predictive VRAM autotune (max {max_batch_size}) | "
            f"resident load ~{load_gb:.1f} GB fits VRAM {vram_gb:.1f} GB"
        )

    return LoadPlan(
        quantization=resolved,
        device_map=device_map,
        max_memory=max_memory,
        abliteration_dtypes=abliteration_dtypes_for(dtype),
        abliteration_batch_size=batch_size,
        abliteration_max_batch_size=max_batch_size,
        estimated_weight_gb=estimated_weight,
        estimated_load_gb=load_gb,
        available_vram_gb=vram_gb,
        available_ram_gb=available,
        commit_limit_gb=commit_limit,
        reasons=reasons,
        warnings=warnings,
    )


def _max_memory(
    *,
    device: str,
    vram_gb: float,
    ram_gb: float,
    available_ram_gb: float,
    commit_limit_gb: float,
    load_gb: float = 0.0,
    quantization: str = "none",
) -> dict[str, str] | None:
    if device == "cpu":
        cpu_budget = _cpu_budget_gb(
            ram_gb,
            available_ram_gb,
            commit_limit_gb,
            vram_gb=0.0,
            load_gb=load_gb,
        )
        return {"cpu": format_memory_budget(cpu_budget)} if cpu_budget else None

    if vram_gb <= 0:
        return None

    # 4-bit that fits in VRAM must stay GPU-only. A cpu= key forces Accelerate
    # to split layers; bitsandbytes then either refuses or crashes on Windows.
    if quantization == "bnb_4bit" and load_gb and load_gb <= vram_gb * 0.9:
        gpu_budget = min(vram_gb - 0.4, max(load_gb + 0.8, vram_gb - 0.8))
        return {"0": format_memory_budget(gpu_budget)}

    gpu_budget = max(1.0, vram_gb - 1.0)
    cpu_budget = _cpu_budget_gb(
        ram_gb,
        available_ram_gb,
        commit_limit_gb,
        vram_gb=vram_gb,
        load_gb=load_gb,
        gpu_budget_gb=gpu_budget,
    )
    mapping: dict[str, str] = {"0": format_memory_budget(gpu_budget)}
    if cpu_budget:
        mapping["cpu"] = format_memory_budget(cpu_budget)
    return mapping


def _cpu_budget_gb(
    ram_gb: float,
    available_ram_gb: float,
    commit_limit_gb: float,
    *,
    vram_gb: float,
    load_gb: float = 0.0,
    gpu_budget_gb: float = 0.0,
) -> float:
    if ram_gb <= 0 and available_ram_gb <= 0 and commit_limit_gb <= 0:
        return 0.0
    from_total = ram_gb * 0.65 if ram_gb else 0.0
    from_available = max(available_ram_gb - 1.5, 0.0) if available_ram_gb else 0.0
    leftover = max(load_gb - gpu_budget_gb + 8.0, 0.0) if load_gb else 0.0
    budget = max(from_total, from_available, leftover, 4.0)
    if commit_limit_gb > 0:
        # Leave room for the OS, GPU mapping, and Python itself.
        budget = min(budget, max(4.0, commit_limit_gb - vram_gb - 8.0))
    return max(budget, 4.0)


def _windows_commit_limit_gb() -> float:
    try:
        import ctypes
        from ctypes import wintypes

        class PerformanceInformation(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("CommitTotal", ctypes.c_size_t),
                ("CommitLimit", ctypes.c_size_t),
                ("CommitPeak", ctypes.c_size_t),
                ("PhysicalTotal", ctypes.c_size_t),
                ("PhysicalAvailable", ctypes.c_size_t),
                ("SystemCache", ctypes.c_size_t),
                ("KernelTotal", ctypes.c_size_t),
                ("KernelPaged", ctypes.c_size_t),
                ("KernelNonpaged", ctypes.c_size_t),
                ("PageSize", ctypes.c_size_t),
                ("HandleCount", wintypes.DWORD),
                ("ProcessCount", wintypes.DWORD),
                ("ThreadCount", wintypes.DWORD),
            ]

        info = PerformanceInformation()
        info.cb = ctypes.sizeof(info)
        if not ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(info), info.cb):
            return 0.0
        return (int(info.CommitLimit) * int(info.PageSize)) / (1024**3)
    except Exception as error:
        logger.debug("Could not read Windows commit limit: %s", error)
        return 0.0


def _linux_commit_limit_gb() -> float:
    path = "/proc/meminfo"
    if not os.path.exists(path):
        return 0.0
    try:
        values: dict[str, int] = {}
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if ":" not in line:
                    continue
                key, raw_value = line.split(":", 1)
                number = raw_value.strip().split()[0]
                if number.isdigit():
                    values[key] = int(number)
        total_kb = values.get("MemTotal", 0) + values.get("SwapTotal", 0)
        return total_kb / (1024**2)
    except OSError as error:
        logger.debug("Could not read %s: %s", path, error)
        return 0.0


def describe_pagefile_fix() -> str:
    """User-facing instructions for Windows error 1455."""
    return (
        "Increase the Windows paging file to at least 64 GB: "
        "Settings → System → About → Advanced system settings → Performance Settings → "
        "Advanced → Virtual memory → Custom size (for example initial 65536 MB, "
        "maximum 131072 MB), then reboot."
    )


def looks_like_pagefile_error(message: str) -> bool:
    lower = message.lower()
    return any(
        marker in lower
        for marker in (
            "os error 1455",
            "error 1455",
            "paging file",
            "page file",
            "подкачки",
            "файл подкачки",
        )
    )


WINDOWS_ACCESS_VIOLATION_CODES = {3221225477, -1073741819}


def looks_like_access_violation(return_code: int | None = None, message: str = "") -> bool:
    if return_code in WINDOWS_ACCESS_VIOLATION_CODES:
        return True
    lower = message.lower()
    return "3221225477" in lower or "0xc0000005" in lower or "access violation" in lower


def looks_like_bnb_offload_error(message: str) -> bool:
    lower = message.lower()
    return "dispatched on the cpu or the disk" in lower or (
        "llm_int8_enable_fp32_cpu_offload" in lower and "quantized" in lower
    )


def looks_like_meta_tensor_error(message: str) -> bool:
    lower = message.lower()
    return "meta tensor" in lower or "cannot be called on meta tensors" in lower


def looks_like_oom_error(message: str) -> bool:
    lower = message.lower()
    return any(
        marker in lower
        for marker in (
            "out of memory",
            "cuda out of memory",
            "not enough memory",
            "failed to allocate",
        )
    )


def compact_process_error(output: str) -> str:
    """Keep the real exception, drop Rich traceback chrome."""
    interesting: list[str] = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line or set(line) <= set("┌┐└┘│─╰╯╭╮━┃"):
            continue
        lowered = line.lower()
        if (
            line.startswith(
                (
                    "Exception:",
                    "RuntimeError:",
                    "OSError:",
                    "ValueError:",
                    "MemoryError:",
                    "NotImplementedError:",
                    "AttributeError:",
                    "TypeError:",
                )
            )
            or "failed (" in lowered
            or "os error" in lowered
            or "out of memory" in lowered
            or "meta tensor" in lowered
        ):
            interesting.append(line)
    return "\n".join(interesting[-4:])


def explain_model_load_failure(output: str, return_code: int) -> str:
    """Turn a Abliteration/Transformers crash into an actionable error."""
    compact = compact_process_error(output)
    if looks_like_access_violation(return_code, output):
        return (
            "Abliteration crashed with Windows access violation 0xC0000005 "
            f"(exit code {return_code}). "
            "On 8 GB laptops this usually means bitsandbytes ran out of RAM/VRAM "
            "while mapping a VL checkpoint (vision + text) or mixing 4-bit GPU "
            "layers with CPU offload. Anlord Abliterator loads the text-only "
            "CausalLM, does not create a parent CUDA context, and will not dump "
            "BF16 onto this GPU. Close Chrome/Discord/other GPU programs and rerun "
            "the same command so the finished baseline is reused. If weight load "
            "still dies around 20-30%, enlarge the Windows pagefile to 64+ GB and reboot."
        )
    if looks_like_meta_tensor_error(output) or looks_like_meta_tensor_error(compact):
        return (
            "Abliteration hit a meta-tensor error because Accelerate offloaded 4-bit "
            "modules to disk. Anlord Abliterator retries with device_map=cuda and "
            "no max_memory so the whole 4-bit model stays on the GPU."
        )
    if looks_like_pagefile_error(output):
        return (
            "Windows pagefile is too small to load this model (os error 1455). "
            + (f"{compact} " if compact else "")
            + describe_pagefile_fix()
        )
    if looks_like_bnb_offload_error(output):
        return (
            "bitsandbytes refused to place 4-bit layers on CPU. Anlord Abliterator now "
            "enables llm_int8_enable_fp32_cpu_offload in the Abliteration bridge so leftover "
            "layers can sit in RAM/pagefile. Update and rerun with --no-resume."
        )
    if looks_like_oom_error(output):
        return (
            "The model does not fit into available GPU/CPU memory. "
            + (f"{compact} " if compact else "")
            + "Use --quantization auto or bnb_4bit, or keep none with device_map=auto "
            "and a large Windows pagefile."
        )
    if compact:
        return f"Abliteration exited with code {return_code}: {compact}"
    return f"Abliteration exited with code {return_code}"


def apply_plan_to_mapping(target: Mapping[str, object] | None, plan: LoadPlan) -> dict[str, object]:
    """Return a shallow copy of *target* with planner fields applied."""
    updated = dict(target or {})
    updated["quantization"] = plan.quantization
    updated["device_map"] = plan.device_map
    updated["max_memory"] = plan.max_memory
    return updated
