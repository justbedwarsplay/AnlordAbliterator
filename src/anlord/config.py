# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Configuration management for Anlord Abliterator pipeline.
Handles settings validation, defaults, and CLI argument processing.
"""

import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional

from .utils.helpers import ensure_dir


# =============================================================================
# Enums
# =============================================================================


class RunMode(str, Enum):
    """Evaluation mode - quick for fast analysis, full for complete evaluation."""

    QUICK = "quick"
    FULL = "full"


class ExportFormat(str, Enum):
    """Output format for reports."""

    HTML = "html"
    JSON = "json"
    CSV = "csv"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark."""

    name: str
    task_id: str
    num_fewshot: int = 0
    description: str = ""
    enabled: bool = True


@dataclass
class HardwareMetrics:
    """Hardware metrics collected during execution."""

    peak_vram_gb: float = 0.0
    peak_ram_gb: float = 0.0
    peak_gpu_utilization: int = 0
    peak_cpu_percent: float = 0.0
    avg_gpu_utilization: float = 0.0
    avg_cpu_percent: float = 0.0
    temperature_celsius: float = 0.0
    duration_seconds: float = 0.0
    tokens_per_second: float = 0.0
    start_time: str = ""
    end_time: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ModelInfo:
    """Information about a model being processed."""

    model_id: str
    model_path: Optional[str] = None
    architecture: Optional[str] = None
    parameter_count: Optional[int] = None
    dtype: Optional[str] = None
    is_abliterated: bool = False
    abliteration_version: Optional[str] = None


# =============================================================================
# Settings Class
# =============================================================================


@dataclass
class Settings:
    """Main settings for Anlord Abliterator pipeline."""

    # Model Configuration
    model: str = "unsloth/gpt-oss-20b-BF16"
    model_commit: Optional[str] = None
    output_dir: Path = field(default_factory=lambda: Path("./output"))
    cache_dir: Path = field(default_factory=lambda: Path("./cache"))
    prefetch_model: bool = True

    # Abliteration Configuration
    abliteration_trials: int = 100
    abliteration_timeout: int = 43200
    abliteration_evaluation_prompts: int = 100
    abliteration_backend: str = "native"  # native | auto
    # Native abliterator overrides
    orthogonalize_direction: bool = True
    row_normalization: str = "full"  # none | pre | full
    winsorization_quantile: float = 1.0
    kl_divergence_scale: float = 1.0
    kl_divergence_target: float = 0.01
    full_normalization_lora_rank: int = 3
    abliteration_subspace_rank: int = 1  # 1=single vector, 3-5=subspace
    abliteration_subspace_method: str = "svd"
    # Include list of model components to ablate (e.g. ["attn.o_proj"] or the
    # prefix ["attn"] to ablate only attention and leave MLP untouched).
    # None = all abliterable components of the model.
    abliteration_components: Optional[list] = None

    # Search acceleration (1.3.0). See docs/optimization_ideas.md.
    # Random-sampling startup trials for TPE (None = automatic, ~min(20, n/8)).
    abliteration_startup_trials: Optional[int] = None
    # Maximum generated tokens when counting refusals during trials.
    abliteration_max_response_length: int = 64
    # Upper bound of the max_weight search range (raise to 2.0+ for
    # attention-only runs so attention alone can carry the ablation).
    abliteration_max_weight_limit: float = 1.5
    # Refusal direction source: "mean" or "median" (per-component median of the
    # residual vectors, robust to massive activations). Recorded in bundles.
    abliteration_direction_source: str = "mean"
    # Early abandonment of provably dominated trials (multi-fidelity evaluation).
    abliteration_pruning: bool = True
    # Enqueue sensible starting configurations when a study is fresh.
    search_seeds: bool = True

    # Staged search (1.4.0): stage 1 optimizes attention only, stage 2
    # optimizes MLP on top of the frozen stage-1 attention winner.
    staged_search: bool = False
    # Fraction of n_trials spent in stage 1 (attention-only).
    staged_stage1_fraction: float = 0.4
    # Silhouette-guided position bounds (experimental).
    silhouette_guided_bounds: bool = False
    capability_proxy: bool = False
    capability_proxy_enabled: bool = False
    capability_proxy_dataset: str = "cais/mmlu"
    capability_proxy_subset: str = "abstract_algebra"
    capability_proxy_samples: int = 20
    capability_proxy_fewshot: int = 5

    # Residual analysis (requires the optional research dependencies)
    print_residual_geometry: bool = False
    plot_residuals: bool = False
    residual_plot_path: str = "plots"
    residual_plot_title: str = 'PaCMAP Projection of Residual Vectors for "Harmless" and "Harmful" Prompts'
    residual_plot_style: str = "dark_background"
    print_debug_information: bool = False

    # Scorer plugins (extensible objectives for the native pipeline).
    # None = default set (refusals keyword-rate + KL divergence, both minimized).
    # Each entry: {"plugin": "<builtin|module.Class|path.py:Class>",
    #              "optimization": "minimize"|"maximize"|"none",
    #              "instance_name": optional}
    scorers: Optional[list] = None
    # Raw settings tables for scorer instances, keyed by "<ClassName>" or
    # "<ClassName>_<instance_name>" (only keys known to the scorer are applied).
    scorer_settings: Optional[dict] = None

    # Export / reproduction
    export_strategy: str = "merge"  # merge | adapter
    max_shard_size: str = "5GB"
    # Which reproduction information to generate at export:
    # "full" (+system info), "basic" (settings + packages), "none".
    reproducibility_information: str = "full"
    # Reproduction mode: path or URL of a reproduce.json file to re-apply.
    reproduce: Optional[str] = None
    # Whether to attempt reproduction even if there are environment mismatches
    # (None = proceed with a warning, False = abort, True = proceed silently).
    ignore_mismatches: Optional[bool] = None

    # Benchmark Configuration
    benchmarks: list[str] = field(
        default_factory=lambda: [
            "mmlu",
            "gsm8k",
            "hellaswag",
            "arc_challenge",
            "winogrande",
            "truthfulqa",
        ]
    )
    mode: RunMode = RunMode.QUICK
    limit: Optional[int] = 100
    num_fewshot: int = 0
    native_benchmarks: bool = True

    # Hardware Configuration
    dtype: str = "auto"
    device: str = "cuda"
    device_map: str = "auto"
    max_memory: Optional[dict] = None
    quantization: str = "auto"
    abliteration_batch_size: Optional[int] = None
    abliteration_max_batch_size: Optional[int] = None
    batch_size: int = 1
    # Pipeline extras (Tasks 3/4)
    baseline_evaluate: bool = False
    yes: bool = False

    # Reproducibility
    seed: int = 42

    # Report Configuration
    report_formats: list[ExportFormat] = field(
        default_factory=lambda: [ExportFormat.HTML, ExportFormat.JSON, ExportFormat.CSV]
    )

    # Pipeline Configuration
    skip_baseline: bool = False
    skip_abliteration: bool = False
    skip_benchmarks: bool = False
    resume: bool = True

    # Logging
    verbose: bool = True
    log_file: Optional[Path] = None

    def __post_init__(self):
        """Normalize and validate user-provided settings."""
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        if isinstance(self.cache_dir, str):
            self.cache_dir = Path(self.cache_dir)
        if isinstance(self.log_file, str):
            self.log_file = Path(self.log_file)

        # Keep explicit 6-task default for backwards compat (tests expect 6)
        _classic_default = ["mmlu", "gsm8k", "hellaswag", "arc_challenge", "winogrande", "truthfulqa"]
        if self.benchmarks is None:
            self.benchmarks = list(_classic_default)
        elif isinstance(self.benchmarks, str):
            self.benchmarks = [b.strip() for b in self.benchmarks.split(",") if b.strip()]
        else:
            self.benchmarks = [str(b).strip() for b in self.benchmarks if str(b).strip()]
        if not self.benchmarks:
            self.benchmarks = list(_classic_default)

        if isinstance(self.mode, str):
            try:
                self.mode = RunMode(self.mode)
            except ValueError as error:
                raise ValueError(f"Unsupported run mode: {self.mode}") from error

        if self.abliteration_backend == "abliteration":
            self.abliteration_backend = "native"
        if self.abliteration_backend not in {"native", "auto"}:
            raise ValueError(f"Unsupported abliteration_backend: {self.abliteration_backend}")
        if self.abliteration_components is not None:
            if isinstance(self.abliteration_components, str):
                self.abliteration_components = [
                    name.strip() for name in self.abliteration_components.split(",") if name.strip()
                ]
            self.abliteration_components = [
                str(name).strip() for name in self.abliteration_components if str(name).strip()
            ]
            if not self.abliteration_components:
                raise ValueError(
                    "abliteration_components cannot be empty; omit it to ablate all components"
                )
        if self.abliteration_startup_trials is not None and (
            self.abliteration_startup_trials < 0
        ):
            raise ValueError("abliteration_startup_trials cannot be negative")
        if self.abliteration_max_response_length < 8:
            raise ValueError("abliteration_max_response_length must be at least 8 tokens")
        self.abliteration_direction_source = str(self.abliteration_direction_source).lower()
        if self.abliteration_direction_source not in {"mean", "median"}:
            raise ValueError(
                f"Unsupported abliteration_direction_source: {self.abliteration_direction_source}"
            )
        if self.abliteration_max_weight_limit <= 0.85:
            raise ValueError(
                "abliteration_max_weight_limit must be above 0.85 (the attention max_weight "
                "search lower bound is 0.8)"
            )
        if self.row_normalization not in {"none", "pre", "full"}:
            raise ValueError(f"Unsupported row_normalization: {self.row_normalization}")
        if self.dtype not in {"auto", "float16", "bfloat16", "float32"}:
            raise ValueError(f"Unsupported dtype: {self.dtype}")
        if self.device not in {"auto", "cuda", "cpu", "mps"}:
            raise ValueError(f"Unsupported device: {self.device}")
        if self.quantization not in {"auto", "none", "bnb_4bit", "bnb_8bit"}:
            raise ValueError(f"Unsupported quantization: {self.quantization}")
        if self.device_map not in {"auto", "cuda", "cpu", "mps"}:
            raise ValueError(f"Unsupported device_map: {self.device_map}")
        self.export_strategy = str(self.export_strategy).lower()
        if self.export_strategy not in {"merge", "adapter"}:
            raise ValueError(f"Unsupported export_strategy: {self.export_strategy}")
        self.reproducibility_information = str(self.reproducibility_information).lower()
        if self.reproducibility_information not in {"full", "basic", "none"}:
            raise ValueError(
                f"Unsupported reproducibility_information: {self.reproducibility_information}"
            )
        if self.scorers is not None:
            if isinstance(self.scorers, str):
                raise ValueError(
                    "scorers must be a list of {plugin, optimization, instance_name} dicts"
                )
            normalized_scorers = []
            for entry in self.scorers:
                if isinstance(entry, str):
                    entry = {"plugin": entry}
                if not isinstance(entry, dict):
                    raise ValueError(
                        "Each scorer entry must be a dict with a 'plugin' key"
                    )
                if "plugin" not in entry:
                    raise ValueError("Each scorer entry requires a 'plugin' key")
                normalized_scorers.append(dict(entry))
            self.scorers = normalized_scorers
        if self.scorer_settings is not None and not isinstance(self.scorer_settings, dict):
            raise ValueError("scorer_settings must be a dict of namespace tables")
        if self.ignore_mismatches is not None and not isinstance(self.ignore_mismatches, bool):
            raise ValueError("ignore_mismatches must be a boolean or None")
        if not 0.05 <= self.staged_stage1_fraction <= 0.95:
            raise ValueError("staged_stage1_fraction must be within [0.05, 0.95]")
        if self.abliteration_trials < 1:
            raise ValueError("abliteration_trials must be at least 1")
        if self.abliteration_timeout < 1:
            raise ValueError("abliteration_timeout must be at least 1 second")
        if self.abliteration_evaluation_prompts < 1:
            raise ValueError("abliteration_evaluation_prompts must be at least 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.abliteration_batch_size is not None and self.abliteration_batch_size < 0:
            raise ValueError("abliteration_batch_size must be 0 (auto) or a positive int")
        if self.limit is not None and self.limit < 1:
            raise ValueError("limit must be at least 1 when specified")
        if self.num_fewshot < 0:
            raise ValueError("num_fewshot cannot be negative")

        try:
            self.report_formats = [
                value if isinstance(value, ExportFormat) else ExportFormat(value)
                for value in self.report_formats
            ]
        except ValueError as error:
            raise ValueError("Unsupported report format") from error

    def get_results_dir(self) -> Path:
        """Get the results directory."""
        return self.output_dir / "results"

    def get_baseline_results_dir(self) -> Path:
        """Get the baseline results directory."""
        return self.get_results_dir() / "baseline"

    def get_abliterated_results_dir(self) -> Path:
        """Get the abliterated results directory."""
        return self.get_results_dir() / "abliterated"

    def get_reports_dir(self) -> Path:
        """Get the reports directory."""
        return self.output_dir / "reports"

    def get_models_dir(self) -> Path:
        """Get the models directory."""
        return self.output_dir / "models"

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        data = asdict(self)
        # Convert Path objects to strings
        for key in ["output_dir", "cache_dir", "log_file"]:
            if isinstance(data.get(key), Path):
                data[key] = str(data[key])
        # Convert enum values
        if isinstance(data.get("mode"), RunMode):
            data["mode"] = data["mode"].value
        if isinstance(data.get("report_formats"), list):
            data["report_formats"] = [
                f.value if isinstance(f, ExportFormat) else f for f in data["report_formats"]
            ]
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Settings":
        """Create Settings from a dictionary without mutating the caller's data."""
        data = data.copy()
        if data.get("abliteration_backend") == "abliteration":
            data["abliteration_backend"] = "native"
        # Convert string paths
        if "output_dir" in data and isinstance(data["output_dir"], str):
            data["output_dir"] = Path(data["output_dir"])
        if "cache_dir" in data and isinstance(data["cache_dir"], str):
            data["cache_dir"] = Path(data["cache_dir"])
        if "log_file" in data and isinstance(data["log_file"], str):
            data["log_file"] = Path(data["log_file"])

        # Parse mode
        if "mode" in data:
            if isinstance(data["mode"], str):
                data["mode"] = RunMode.QUICK if data["mode"] == "quick" else RunMode.FULL

        # Parse report formats
        if "report_formats" in data:
            formats = []
            for f in data["report_formats"]:
                if isinstance(f, str):
                    formats.append(ExportFormat(f))
                else:
                    formats.append(f)
            data["report_formats"] = formats

        return cls(**data)

    def save(self, path: Path) -> None:
        """Save settings to JSON file."""
        path = Path(path)
        ensure_dir(path.parent)
        with open(path, "w", encoding="utf-8") as file:
            json.dump(self.to_dict(), file, indent=2)

    @classmethod
    def load(cls, path: Path) -> "Settings":
        """Load settings from JSON file."""
        path = Path(path)
        with open(path, encoding="utf-8") as file:
            data = json.load(file)
        return cls.from_dict(data)


# =============================================================================
# Benchmark Definitions
# =============================================================================

AVAILABLE_BENCHMARKS: dict[str, BenchmarkConfig] = {
    "mmlu": BenchmarkConfig(
        name="MMLU",
        task_id="mmlu",
        num_fewshot=5,
        description="Massively Multilingual Language Understanding",
    ),
    "gsm8k": BenchmarkConfig(
        name="GSM8K", task_id="gsm8k", num_fewshot=5, description="Grade School Math 8K"
    ),
    "hellaswag": BenchmarkConfig(
        name="HellaSwag",
        task_id="hellaswag",
        num_fewshot=10,
        description="Commonsense inference challenge",
    ),
    "arc_challenge": BenchmarkConfig(
        name="ARC-Challenge",
        task_id="arc_challenge",
        num_fewshot=0,
        description="AI2 Reasoning Challenge",
    ),
    "winogrande": BenchmarkConfig(
        name="Winogrande",
        task_id="winogrande",
        num_fewshot=0,
        description="WinoGrande commonsense reasoning",
    ),
    "truthfulqa": BenchmarkConfig(
        name="TruthfulQA",
        task_id="truthfulqa",
        num_fewshot=0,
        description="Truthful question answering",
    ),
    "gpqa": BenchmarkConfig(name="GPQA Diamond", task_id="gpqa_diamond", num_fewshot=0, description="GPQA Diamond (graduate science QA, hardest)"),
    "gpqa_diamond": BenchmarkConfig(name="GPQA Diamond", task_id="gpqa_diamond", num_fewshot=0, description="GPQA Diamond"),
    "gpqa_main": BenchmarkConfig(name="GPQA Main", task_id="gpqa_main", num_fewshot=0, description="GPQA main"),
    "gpqa_extended": BenchmarkConfig(name="GPQA Extended", task_id="gpqa_extended", num_fewshot=0, description="GPQA extended"),
    "mmlu_pro": BenchmarkConfig(name="MMLU-Pro", task_id="mmlu_pro", num_fewshot=5, description="MMLU-Pro 10-option"),
    "ifstruct": BenchmarkConfig(name="Ifstruct V1", task_id="ifstruct", num_fewshot=0, description="LiquidAI Ifstruct instruction following"),
    "ifstruct_v1": BenchmarkConfig(name="Ifstruct V1", task_id="ifstruct", num_fewshot=0, description="LiquidAI Ifstruct"),
    "parsebench": BenchmarkConfig(name="ParseBench", task_id="parsebench", num_fewshot=0, description="ParseBench document parsing"),
    "extractbench": BenchmarkConfig(name="ExtractBench", task_id="extractbench", num_fewshot=0, description="ExtractBench structured extraction"),
    "screenspot_pro": BenchmarkConfig(name="ScreenSpot-Pro", task_id="screenspot_pro", num_fewshot=0, description="ScreenSpot-Pro GUI grounding"),
    "mmmu_pro": BenchmarkConfig(name="MMMU-Pro", task_id="mmmu_pro", num_fewshot=0, description="MMMU-Pro vision reasoning"),
}


def get_benchmark_configs(benchmark_names: list[str]) -> list[BenchmarkConfig]:
    """Get benchmark configurations for the specified benchmark names."""
    configs = []
    for name in benchmark_names:
        if name in AVAILABLE_BENCHMARKS:
            configs.append(AVAILABLE_BENCHMARKS[name])
        else:
            # Create a generic config for unknown benchmarks
            configs.append(
                BenchmarkConfig(
                    name=name.upper(), task_id=name, description=f"User-specified benchmark: {name}"
                )
            )
    return configs
