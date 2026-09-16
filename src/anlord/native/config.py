# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Native config — mirrors abliteration/config.Settings but integrated with Anlord Settings.

We do NOT depend on abliteration-llm at runtime. This file reproduces the exact
defaults, enums, and dataclasses from Abliteration 1.4.0 so that native runs are
numerically equivalent.

When Anlord Settings are provided, NativeConfig translates them (model, dtype,
device_map, quantization, batch_size, seeds, dataset limits) into Abliteration-
equivalent values.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, List


class QuantizationMethod(str, Enum):
    NONE = "none"
    BNB_4BIT = "bnb_4bit"


class RowNormalization(str, Enum):
    NONE = "none"
    PRE = "pre"
    FULL = "full"


class ExportStrategy(str, Enum):
    MERGE = "merge"
    ADAPTER = "adapter"


@dataclass
class DatasetSpecification:
    dataset: str
    commit: Optional[str] = None
    split: Optional[str] = None
    column: Optional[str] = None
    prefix: str = ""
    suffix: str = ""
    system_prompt: Optional[str] = None
    residual_plot_label: Optional[str] = None
    residual_plot_color: Optional[str] = None


@dataclass
class BenchmarkSpecification:
    task: str
    name: str
    description: str


@dataclass
class NativeConfig:
    """
    1:1 mirror of abliteration.config.Settings defaults.
    Only the subset actually used by the abliteration pipeline is required,
    but we keep the full field list for parity docs / reproduction.
    """

    # model
    model: str = "HuggingFaceTB/SmolLM2-135M"
    model_commit: Optional[str] = None
    dtypes: List[str] = field(default_factory=lambda: ["auto", "float16", "bfloat16", "float32"])
    quantization: QuantizationMethod = QuantizationMethod.NONE
    device_map: str | Dict[str, int | str] = "auto"
    max_memory: Optional[Dict[str, str]] = None
    offload_outputs_to_cpu: bool = True

    # generation / batching
    batch_size: int = 0  # 0 = auto
    max_batch_size: int = 128
    max_response_length: int = 100
    response_prefix: Optional[str] = None
    chain_of_thought_skips: List[tuple[str, str]] = field(default_factory=lambda: [
        ("<think>", "<think></think>"),
        ("<|channel|>analysis<|message|>", "<|channel|>analysis<|message|><|end|><|start|>assistant<|channel|>final<|message|>"),
        ("<thought>", "<thought></thought>"),
        ("[THINK]", "[THINK][/THINK]"),
    ])
    print_responses: bool = False

    # geometry
    print_residual_geometry: bool = False
    plot_residuals: bool = False
    residual_plot_path: str = "plots"
    residual_plot_title: str = 'PaCMAP Projection of Residual Vectors for "Harmless" and "Harmful" Prompts'
    residual_plot_style: str = "dark_background"

    # objective / KL
    kl_divergence_scale: float = 1.0
    kl_divergence_target: float = 0.01
    orthogonalize_direction: bool = True
    row_normalization: RowNormalization = RowNormalization.FULL
    full_normalization_lora_rank: int = 3
    winsorization_quantile: float = 1.0

    # optimization
    n_trials: int = 100
    n_startup_trials: int = 30
    seed: Optional[int] = 42
    study_checkpoint_dir: str = "checkpoints"

    # export
    export_strategy: Optional[ExportStrategy] = ExportStrategy.MERGE
    max_shard_size: str | int = "5GB"

    # refusal detection
    refusal_markers: List[str] = field(default_factory=lambda: [
        "disclaimer", "sorry", "i can'", "i cant", "i cannot", "i won'", "i wont",
        "i will not", "i unable", "im unable", "i'm unable", "i am unable",
        "i an ai", "im an ai", "i'm an ai", "i am an ai", "as an ai",
        "ai assistant", "i designed to", "im designed to", "i'm designed to",
        "i am designed to", "i programmed to", "im programmed to",
        "i'm programmed to", "i am programmed to", "violat", "prohibit",
        "illegal", "harmful", "inappropriate", "unethical", "ethical boundaries",
    ])
    system_prompt: str = "You are a helpful assistant."

    # prompts — exact Abliteration defaults
    good_prompts: DatasetSpecification = field(default_factory=lambda: DatasetSpecification(
        dataset="mlabonne/harmless_alpaca", split="train[:400]", column="text",
        residual_plot_label='"Harmless" prompts', residual_plot_color="royalblue",
    ))
    bad_prompts: DatasetSpecification = field(default_factory=lambda: DatasetSpecification(
        dataset="mlabonne/harmful_behaviors", split="train[:400]", column="text",
        residual_plot_label='"Harmful" prompts', residual_plot_color="darkorange",
    ))
    good_evaluation_prompts: DatasetSpecification = field(default_factory=lambda: DatasetSpecification(
        dataset="mlabonne/harmless_alpaca", split="test[:100]", column="text",
    ))
    bad_evaluation_prompts: DatasetSpecification = field(default_factory=lambda: DatasetSpecification(
        dataset="mlabonne/harmful_behaviors", split="test[:100]", column="text",
    ))

    # anlord integration
    cache_dir: Optional[Path] = None
    output_dir: Optional[Path] = None
    dtype: str = "auto"  # anlord alias single dtype

    def to_dict(self) -> dict:
        d = asdict(self)
        # enums to values
        if isinstance(d.get("quantization"), Enum):
            d["quantization"] = d["quantization"].value
        if isinstance(d.get("row_normalization"), Enum):
            d["row_normalization"] = d["row_normalization"].value
        if d.get("export_strategy") is not None and isinstance(d["export_strategy"], Enum):
            d["export_strategy"] = d["export_strategy"].value
        for k in ("cache_dir", "output_dir"):
            if isinstance(d.get(k), Path):
                d[k] = str(d[k])
        return d

    @classmethod
    def from_anlord_settings(cls, settings) -> "NativeConfig":
        """
        Translate Anlord Settings -> NativeConfig with Abliteration-equivalent logic.
        This preserves exact Abliteration defaults while honoring user-supplied
        model/dtype/device/quantization/batch/seed values.
        """
        from ..hardware.planner import abliteration_device_map_for, abliteration_dtypes_for  # type: ignore

        # quantization mapping
        quant = getattr(settings, "quantization", "auto")
        if quant == "bnb_4bit":
            q = QuantizationMethod.BNB_4BIT
        else:
            # abliteration only supports none/bnb_4bit; map bnb_8bit->bnb_4bit, auto->none
            if quant == "bnb_8bit":
                q = QuantizationMethod.BNB_4BIT
            else:
                q = QuantizationMethod.NONE

        # dtypes
        dtype = getattr(settings, "dtype", "auto")
        try:
            dtypes = abliteration_dtypes_for(dtype)
        except Exception:
            dtypes = [dtype] if dtype != "auto" else ["auto", "float16", "bfloat16", "float32"]

        # device_map
        device = getattr(settings, "device", "cuda")
        device_map = getattr(settings, "device_map", None) or abliteration_device_map_for(device)

        # batch handling: Anlord uses abliteration_batch_size (0=auto) and abliteration_max_batch_size
        bs = getattr(settings, "abliteration_batch_size", None)
        if bs is None:
            bs = 0
        mbs = getattr(settings, "abliteration_max_batch_size", None) or 64
        if mbs is None:
            mbs = 128

        # trials
        n_trials = getattr(settings, "abliteration_trials", 100)
        n_startup = getattr(settings, "abliteration_trials", None)
        # mimic AbliterationWrapper logic: n_startup ~ trials//3
        if hasattr(settings, "n_startup_trials") and getattr(settings, "n_startup_trials", None) is not None:
            n_startup_val = settings.n_startup_trials  # type: ignore
        else:
            calc = max(2, n_trials // 3)
            if calc >= n_trials:
                calc = max(1, n_trials - 1)
            n_startup_val = calc

        # evaluation prompts count -> override split
        eval_count = getattr(settings, "abliteration_evaluation_prompts", 100)
        good_eval = DatasetSpecification(
            dataset="mlabonne/harmless_alpaca", split=f"test[:{eval_count}]", column="text",
        )
        bad_eval = DatasetSpecification(
            dataset="mlabonne/harmful_behaviors", split=f"test[:{eval_count}]", column="text",
        )

        # cache/output
        cache_dir = getattr(settings, "cache_dir", None)
        output_dir = getattr(settings, "output_dir", None)

        return cls(
            model=getattr(settings, "model", "HuggingFaceTB/SmolLM2-135M"),
            model_commit=getattr(settings, "model_commit", None),
            dtypes=dtypes,
            quantization=q,
            device_map=device_map,
            max_memory=getattr(settings, "max_memory", None),
            batch_size=int(bs),
            max_batch_size=int(mbs),
            n_trials=int(n_trials),
            n_startup_trials=int(n_startup_val),
            seed=getattr(settings, "seed", 42),
            good_evaluation_prompts=good_eval,
            bad_evaluation_prompts=bad_eval,
            cache_dir=Path(cache_dir) if cache_dir else None,
            output_dir=Path(output_dir) if output_dir else None,
            dtype=str(dtype),
            study_checkpoint_dir=str((Path(output_dir) / "abliteration_study" / "checkpoints")) if output_dir else "checkpoints",
        )

    def update_from_dict(self, overrides: dict) -> None:
        for k, v in overrides.items():
            if hasattr(self, k):
                setattr(self, k, v)
