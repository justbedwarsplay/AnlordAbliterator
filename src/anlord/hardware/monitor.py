# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Hardware monitoring for Anlord Abliterator pipeline.
Provides real-time monitoring of GPU, VRAM, RAM, CPU, and temperature.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from threading import Thread
from typing import Optional

import psutil
import torch

logger = logging.getLogger(__name__)


@dataclass
class SystemInfo:
    """System information snapshot."""

    # GPU Information
    has_cuda: bool = False
    cuda_version: str = ""
    gpu_name: str = ""
    gpu_count: int = 0
    total_vram_gb: float = 0.0

    # System Information
    cpu_count: int = 0
    cpu_name: str = ""
    total_ram_gb: float = 0.0
    os_name: str = ""

    # Library Versions
    pytorch_version: str = ""
    transformers_version: str = ""
    heretic_version: str = ""
    lm_eval_version: str = ""

    # Python Information
    python_version: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "has_cuda": self.has_cuda,
            "cuda_version": self.cuda_version,
            "gpu_name": self.gpu_name,
            "gpu_count": self.gpu_count,
            "total_vram_gb": self.total_vram_gb,
            "cpu_count": self.cpu_count,
            "cpu_name": self.cpu_name,
            "total_ram_gb": self.total_ram_gb,
            "os_name": self.os_name,
            "pytorch_version": self.pytorch_version,
            "transformers_version": self.transformers_version,
            "heretic_version": self.heretic_version,
            "lm_eval_version": self.lm_eval_version,
            "python_version": self.python_version,
        }


@dataclass
class MetricsSnapshot:
    """A single metrics snapshot."""

    timestamp: float
    vram_used_gb: float = 0.0
    vram_total_gb: float = 0.0
    ram_used_gb: float = 0.0
    ram_total_gb: float = 0.0
    gpu_utilization: float = 0.0
    cpu_percent: float = 0.0
    temperature_celsius: float = 0.0


@dataclass
class HardwareMetrics:
    """Aggregated hardware metrics for a run."""

    peak_vram_gb: float = 0.0
    peak_ram_gb: float = 0.0
    peak_gpu_utilization: float = 0.0
    peak_cpu_percent: float = 0.0
    avg_gpu_utilization: float = 0.0
    avg_cpu_percent: float = 0.0
    peak_temperature_celsius: float = 0.0
    duration_seconds: float = 0.0
    tokens_per_second: float = 0.0
    start_time: str = ""
    end_time: str = ""
    samples: list = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "peak_vram_gb": round(self.peak_vram_gb, 2),
            "peak_ram_gb": round(self.peak_ram_gb, 2),
            "peak_gpu_utilization": round(self.peak_gpu_utilization, 1),
            "peak_cpu_percent": round(self.peak_cpu_percent, 1),
            "avg_gpu_utilization": round(self.avg_gpu_utilization, 1),
            "avg_cpu_percent": round(self.avg_cpu_percent, 1),
            "peak_temperature_celsius": round(self.peak_temperature_celsius, 1),
            "duration_seconds": round(self.duration_seconds, 1),
            "tokens_per_second": round(self.tokens_per_second, 2),
            "start_time": self.start_time,
            "end_time": self.end_time,
        }


class HardwareMonitor:
    """
    Real-time hardware monitoring for the pipeline.

    Tracks GPU utilization, VRAM usage, RAM usage, CPU usage,
    and temperature during pipeline execution.
    """

    def __init__(self, interval: float = 1.0):
        """
        Initialize the hardware monitor.

        Args:
            interval: Sampling interval in seconds
        """
        self.interval = interval
        self._monitoring = False
        self._thread: Optional[Thread] = None
        self._snapshots: list[MetricsSnapshot] = []
        self._start_time: float = 0.0
        self._end_time: float = 0.0
        self._nvml_initialized = False
        self._nvml_handle = None

        # Try to initialize NVML for detailed GPU metrics
        self._init_nvml()

    def _init_nvml(self):
        """Initialize NVIDIA Management Library for GPU monitoring."""
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml_initialized = True
            self._nvml = pynvml

            # Get handle for first GPU
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception as e:
            logger.debug(f"NVML not available: {e}")
            self._nvml_initialized = False

    def _get_vram_info(self) -> tuple[float, float]:
        """Get device-wide VRAM usage in GB.

        Heretic runs in a child process, so ``torch.cuda.memory_reserved()`` in
        the Anlord Abliterator process cannot see its allocations.  NVML reports usage for
        the whole device and therefore gives the correct pipeline peak.
        """
        if self._nvml_initialized and self._nvml_handle:
            try:
                memory = self._nvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                return memory.used / (1024**3), memory.total / (1024**3)
            except Exception:
                pass

        initialized = getattr(torch.cuda, "is_initialized", None)
        if callable(initialized) and not initialized():
            return 0.0, 0.0
        if not torch.cuda.is_available():
            return 0.0, 0.0
        try:
            reserved = torch.cuda.memory_reserved() / (1024**3)
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            return reserved, total
        except Exception:
            return 0.0, 0.0

    def _get_gpu_utilization(self) -> float:
        """Get GPU utilization percentage."""
        if self._nvml_initialized and self._nvml_handle:
            try:
                util = self._nvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                return util.gpu
            except Exception:
                pass
        return 0.0

    def _get_temperature(self) -> float:
        """Get GPU temperature in Celsius."""
        if self._nvml_initialized and self._nvml_handle:
            try:
                temp = self._nvml.nvmlDeviceGetTemperature(
                    self._nvml_handle, self._nvml.NVML_TEMPERATURE_GPU
                )
                return float(temp)
            except Exception:
                pass
        return 0.0

    def _collect_snapshot(self) -> MetricsSnapshot:
        """Collect a single snapshot of hardware metrics."""
        vram_used, vram_total = self._get_vram_info()
        ram = psutil.virtual_memory()

        return MetricsSnapshot(
            timestamp=time.time(),
            vram_used_gb=vram_used,
            vram_total_gb=vram_total,
            ram_used_gb=ram.used / (1024**3),
            ram_total_gb=ram.total / (1024**3),
            gpu_utilization=self._get_gpu_utilization(),
            cpu_percent=psutil.cpu_percent(interval=0.1),
            temperature_celsius=self._get_temperature(),
        )

    def _monitoring_loop(self):
        """Background monitoring loop."""
        while self._monitoring:
            snapshot = self._collect_snapshot()
            self._snapshots.append(snapshot)

            # Update peak metrics
            self._peak_vram = max(self._peak_vram, snapshot.vram_used_gb)
            self._peak_ram = max(self._peak_ram, snapshot.ram_used_gb)
            self._peak_gpu_util = max(self._peak_gpu_util, snapshot.gpu_utilization)
            self._peak_cpu = max(self._peak_cpu, snapshot.cpu_percent)
            self._peak_temp = max(self._peak_temp, snapshot.temperature_celsius)

            time.sleep(self.interval)

    def start(self):
        """Start hardware monitoring."""
        if self._monitoring:
            return

        self._monitoring = True
        self._snapshots = []
        self._start_time = time.time()
        self._peak_vram = 0.0
        self._peak_ram = 0.0
        self._peak_gpu_util = 0.0
        self._peak_cpu = 0.0
        self._peak_temp = 0.0

        self._thread = Thread(target=self._monitoring_loop, daemon=True)
        self._thread.start()
        logger.info("Hardware monitoring started")

    def stop(self) -> HardwareMetrics:
        """
        Stop hardware monitoring and return aggregated metrics.

        Returns:
            HardwareMetrics with peak and average values
        """
        if not self._monitoring:
            return HardwareMetrics()

        self._monitoring = False
        self._end_time = time.time()

        if self._thread:
            self._thread.join(timeout=5.0)

        # Calculate averages
        if self._snapshots:
            avg_gpu = sum(s.gpu_utilization for s in self._snapshots) / len(self._snapshots)
            avg_cpu = sum(s.cpu_percent for s in self._snapshots) / len(self._snapshots)
        else:
            avg_gpu = avg_cpu = 0.0

        metrics = HardwareMetrics(
            peak_vram_gb=self._peak_vram,
            peak_ram_gb=self._peak_ram,
            peak_gpu_utilization=self._peak_gpu_util,
            peak_cpu_percent=self._peak_cpu,
            avg_gpu_utilization=avg_gpu,
            avg_cpu_percent=avg_cpu,
            peak_temperature_celsius=self._peak_temp,
            duration_seconds=self._end_time - self._start_time,
            start_time=datetime.fromtimestamp(self._start_time).isoformat(),
            end_time=datetime.fromtimestamp(self._end_time).isoformat(),
            samples=[s.__dict__ for s in self._snapshots],
        )

        logger.info(f"Hardware monitoring stopped. Duration: {metrics.duration_seconds:.1f}s")
        return metrics

    def __del__(self):
        """Cleanup NVML on deletion."""
        if self._nvml_initialized:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass


def _gpu_info_from_nvml() -> tuple[bool, str, int, float] | None:
    """Read GPU name/VRAM via NVML so the parent process does not create a CUDA context."""
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            count = int(pynvml.nvmlDeviceGetCount())
            if count <= 0:
                return False, "", 0, 0.0
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return True, str(name), count, memory.total / (1024**3)
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        return None


def get_system_info() -> SystemInfo:
    """
    Get comprehensive system information.

    Returns:
        SystemInfo with all available system details
    """
    info = SystemInfo()

    # Python version
    import sys

    info.python_version = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )

    # PyTorch version. torch.version.cuda is a compile-time string and does not
    # create a CUDA context in this parent process.
    info.pytorch_version = torch.__version__
    info.cuda_version = torch.version.cuda or ""

    nvml_gpu = _gpu_info_from_nvml()
    if nvml_gpu is not None:
        info.has_cuda, info.gpu_name, info.gpu_count, info.total_vram_gb = nvml_gpu
    else:
        # Last resort: this initializes a CUDA context and steals VRAM from Heretic.
        info.has_cuda = torch.cuda.is_available()
        if info.has_cuda:
            info.gpu_count = torch.cuda.device_count()
            if info.gpu_count > 0:
                props = torch.cuda.get_device_properties(0)
                info.gpu_name = props.name
                info.total_vram_gb = props.total_memory / (1024**3)

    # System info
    info.cpu_count = psutil.cpu_count(logical=True)
    info.total_ram_gb = psutil.virtual_memory().total / (1024**3)

    # OS
    import platform

    info.os_name = f"{platform.system()} {platform.release()}"

    # Try to get library versions
    try:
        import transformers

        info.transformers_version = transformers.__version__
    except ImportError:
        info.transformers_version = "not installed"

    try:
        import heretic

        info.heretic_version = getattr(heretic, "__version__", "unknown")
    except ImportError:
        info.heretic_version = "not installed"

    try:
        import lm_eval

        info.lm_eval_version = getattr(lm_eval, "__version__", "unknown")
    except ImportError:
        info.lm_eval_version = "not installed"

    # Try to get CPU name on Linux
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if line.startswith("model name"):
                    info.cpu_name = line.split(":")[1].strip()
                    break
    except Exception:
        pass

    return info


def print_system_info(info: SystemInfo):
    """Print system information in a formatted way."""
    print("\n" + "=" * 50)
    print("SYSTEM INFORMATION")
    print("=" * 50)

    print("\n[GPU]")
    if info.has_cuda:
        print(f"  Name:      {info.gpu_name}")
        print(f"  Count:     {info.gpu_count}")
        print(f"  VRAM:      {info.total_vram_gb:.1f} GB")
        print(f"  CUDA:      {info.cuda_version}")
    else:
        print("  CUDA:      Not available")

    print("\n[CPU]")
    print(f"  Count:     {info.cpu_count}")
    if info.cpu_name:
        print(f"  Model:     {info.cpu_name}")

    print("\n[RAM]")
    print(f"  Total:     {info.total_ram_gb:.1f} GB")

    print("\n[OS]")
    print(f"  System:    {info.os_name}")

    print("\n[Libraries]")
    print(f"  Python:    {info.python_version}")
    print(f"  PyTorch:   {info.pytorch_version}")
    print(f"  Transformers: {info.transformers_version}")
    print(f"  Heretic:   {info.heretic_version}")
    print(f"  lm-eval:   {info.lm_eval_version}")

    print("\n" + "=" * 50 + "\n")
