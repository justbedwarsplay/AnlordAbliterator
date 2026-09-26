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

from dataclasses import dataclass, field, asdict, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Dict, List


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
    config: Optional[str] = None
    split: Optional[str] = None
    column: Optional[str] = None
    prefix: str = ""
    suffix: str = ""
    system_prompt: Optional[str] = None
    residual_plot_label: Optional[str] = None
    residual_plot_color: Optional[str] = None


@dataclass
class ScorerConfig:
    """
    Configuration for a single scorer plugin.

    The `plugin` is either a built-in reference such as
    "anlord.scorers.keyword_rate.KeywordRate", a fully-qualified import path,
    or a filesystem reference of the form "path/to/plugin.py:ClassName".

    `optimization` is "minimize" / "maximize" to include the scorer as an
    optimization objective, or "none" to compute the score without optimizing
    for it. `instance_name` distinguishes multiple instances of the same
    plugin class; instance-specific settings then live in `scorer_settings`
    under the key "<ClassName>_<instance_name>".
    """

    plugin: str = "anlord.scorers.keyword_rate.KeywordRate"
    optimization: str = "minimize"  # minimize | maximize | none
    instance_name: Optional[str] = None

    def validate(self) -> None:
        if self.optimization not in {"minimize", "maximize", "none"}:
            raise ValueError(
                f"Unsupported scorer optimization: {self.optimization!r} "
                '(expected "minimize", "maximize" or "none")'
            )
        if self.instance_name is not None:
            name = self.instance_name
            if not name.strip() or "." in name or any(c.isspace() for c in name):
                raise ValueError(
                    f"Invalid scorer instance_name: {name!r} "
                    "(must be non-empty, without dots or whitespace)"
                )


def default_scorer_configs() -> List[ScorerConfig]:
    """Default scorer set: refusals (keyword rate) and KL divergence, both minimized."""
    return [
        ScorerConfig(
            plugin="anlord.scorers.keyword_rate.KeywordRate",
            optimization="minimize",
        ),
        ScorerConfig(
            plugin="anlord.scorers.kl_divergence.KLDivergence",
            optimization="minimize",
        ),
    ]


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
    max_batch_size: int = 256
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
    # Note: kl_divergence_scale / kl_divergence_target are legacy fields kept for
    # config compatibility; with the scorer-plugin pipeline the objective values
    # are the raw scorer values (see anlord/native/scorer.py).
    kl_divergence_scale: float = 1.0
    kl_divergence_target: float = 0.01
    orthogonalize_direction: bool = True
    row_normalization: RowNormalization = RowNormalization.FULL
    full_normalization_lora_rank: int = 3
    winsorization_quantile: float = 1.0

    # scorers (extensible objective system)
    scorers: List[ScorerConfig] = field(default_factory=default_scorer_configs)
    # Raw settings tables for scorer instances, keyed by "<ClassName>" or
    # "<ClassName>_<instance_name>". Only keys known to the scorer's settings
    # schema are applied.
    scorer_settings: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # search acceleration (1.3.0)
    # Upper bound of the max_weight search range. The 1.5 default matches the
    # joint (attention+MLP) pipeline; attention-only runs benefit from a higher
    # limit (e.g. 2.0) because attention alone has to carry the whole ablation.
    max_weight_limit: float = 1.5
    # How the per-layer refusal direction is derived from the residual vectors:
    # "mean" (classic) or "median" (per-component median, robust to massive
    # activations). Run-identity: recorded in reproduction bundles.
    direction_source: str = "mean"
    # Early abandonment of provably dominated trials: evaluate the cheap KL
    # forward first, then refusal counts on batch-aligned prefixes of the
    # evaluation set, and prune as soon as some completed trial is guaranteed
    # to dominate this one. See docs/optimization_ideas.md.
    evaluation_pruning: bool = True
    pruning_fractions: List[float] = field(default_factory=lambda: [0.25, 0.5])
    # Enqueue a few sensible starting configurations when a study is fresh, so
    # the Pareto front forms immediately instead of after random exploration.
    search_seeds: bool = True

    # debug / reproducibility metadata
    print_debug_information: bool = False
    # Which reproduction information to generate at export: "full" (settings,
    # package versions and system information), "basic" (settings and package
    # versions), or "none".
    reproducibility_information: str = "full"
    # Reproduction mode: whether to attempt reproduction even if there are
    # environment mismatches (None = proceed with a warning in the
    # non-interactive pipeline, False = abort, True = proceed silently).
    ignore_mismatches: Optional[bool] = None

    # subspace (feature 1): multi-vector refusal subspace via SVD
    refusal_subspace_rank: int = 1  # 1=legacy single vector, 3-5=subspace
    refusal_subspace_method: str = "svd"

    # Include list of model components to ablate (e.g. ["attn.o_proj"] to ablate
    # only attention and leave MLP untouched). None = all abliterable components.
    # Names are matched exactly or by prefix ("attn" matches "attn.o_proj").
    abliteration_components: Optional[List[str]] = None

    # Staged search (1.4.0): stage 1 optimizes attention components only
    # (MLP frozen at identity), stage 2 freezes the stage-1 attention winner
    # (with a +/-20% max_weight rescale) and optimizes the MLP components.
    # Requires both attention and MLP components; auto-disables otherwise.
    staged_search: bool = False
    # Fraction of n_trials spent in stage 1 (attention-only).
    staged_stage1_fraction: float = 0.4
    # Silhouette-guided bounds (1.4.0): compute per-layer silhouette scores of
    # the good/bad residual clusters and lower-bound the max_weight_position
    # search range by the first layer with meaningful cluster separation.
    silhouette_guided_bounds: bool = False

    # capability proxy (feature 2): 3rd objective to preserve MMLU
    capability_proxy_enabled: bool = False
    capability_proxy_dataset: str = "cais/mmlu"
    capability_proxy_subset: str = "abstract_algebra"
    capability_proxy_samples: int = 20
    capability_proxy_fewshot: int = 5

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
        mbs = getattr(settings, "abliteration_max_batch_size", None) or 256
        if mbs is None:
            mbs = 256

        # trials
        n_trials = getattr(settings, "abliteration_trials", 100)
        # Random-sampling startup trials: multivariate TPE models well after a
        # couple dozen observations, so a third of the budget (the old rule)
        # was wasted. Explicit user value always wins.
        explicit_startup = getattr(settings, "abliteration_startup_trials", None)
        if explicit_startup is not None:
            n_startup_val = int(explicit_startup)
        else:
            n_startup_val = min(20, max(8, n_trials // 8))
            if n_startup_val >= n_trials:
                n_startup_val = max(1, n_trials - 1)

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

        # residual analysis (default plot location lives inside the output dir)
        residual_plot_path = str(getattr(settings, "residual_plot_path", "plots") or "plots")
        if residual_plot_path == "plots" and output_dir:
            residual_plot_path = str(Path(output_dir) / "plots")

        # scorer plugins
        raw_scorers = getattr(settings, "scorers", None)
        if raw_scorers:
            scorers: List[ScorerConfig] = []
            for entry in raw_scorers:
                if isinstance(entry, ScorerConfig):
                    scorer_cfg = entry
                elif isinstance(entry, dict):
                    scorer_cfg = ScorerConfig(**entry)
                else:
                    scorer_cfg = ScorerConfig(plugin=str(entry))
                scorer_cfg.validate()
                scorers.append(scorer_cfg)
        else:
            scorers = default_scorer_configs()

        raw_scorer_settings = getattr(settings, "scorer_settings", None) or {}
        scorer_settings: Dict[str, Dict[str, Any]] = {
            str(namespace): dict(table or {})
            for namespace, table in raw_scorer_settings.items()
        }

        # Keep the classic evaluation-prompt-count knob working for the built-in
        # scorers: inject their prompt specifications unless the user already
        # configured them in scorer_settings.
        for scorer_cfg in scorers:
            if not scorer_cfg.plugin.startswith("anlord."):
                continue
            class_name = scorer_cfg.plugin.rsplit(".", 1)[-1]
            if class_name not in ("KeywordRate", "KLDivergence"):
                continue
            dataset = (
                "mlabonne/harmful_behaviors"
                if class_name == "KeywordRate"
                else "mlabonne/harmless_alpaca"
            )
            prompts_default = {
                "dataset": dataset,
                "split": f"test[:{eval_count}]",
                "column": "text",
            }
            table = scorer_settings.get(class_name)
            if table is None:
                scorer_settings[class_name] = {"prompts": prompts_default}
            elif "prompts" not in table:
                scorer_settings[class_name] = {**table, "prompts": prompts_default}

        # export strategy
        raw_export = getattr(settings, "export_strategy", None) or "merge"
        export_strategy = ExportStrategy(str(raw_export).lower())

        raw_repro_info = str(getattr(settings, "reproducibility_information", "full") or "full").lower()
        if raw_repro_info not in {"full", "basic", "none"}:
            raise ValueError(f"Unsupported reproducibility_information: {raw_repro_info}")

        direction_source = str(getattr(settings, "abliteration_direction_source", "mean") or "mean").lower()
        if direction_source not in {"mean", "median"}:
            raise ValueError(f"Unsupported direction_source: {direction_source}")
        max_weight_limit = float(getattr(settings, "abliteration_max_weight_limit", 1.5) or 1.5)
        if max_weight_limit <= 0.85:
            raise ValueError(
                "abliteration_max_weight_limit must be above 0.85 (the attention max_weight "
                "search lower bound is 0.8)"
            )

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
            refusal_subspace_rank=int(getattr(settings, "abliteration_subspace_rank", 1) or 1),
            refusal_subspace_method=str(getattr(settings, "abliteration_subspace_method", "svd")),
            abliteration_components=(
                [str(name) for name in settings.abliteration_components]
                if getattr(settings, "abliteration_components", None)
                else None
            ),
            staged_search=bool(getattr(settings, "staged_search", False)),
            staged_stage1_fraction=float(getattr(settings, "staged_stage1_fraction", 0.4) or 0.4),
            silhouette_guided_bounds=bool(getattr(settings, "silhouette_guided_bounds", False)),
            capability_proxy_enabled=bool(getattr(settings, "capability_proxy_enabled", False) or getattr(settings, "capability_proxy", False)),
            capability_proxy_dataset=str(getattr(settings, "capability_proxy_dataset", "cais/mmlu")),
            capability_proxy_subset=str(getattr(settings, "capability_proxy_subset", "abstract_algebra")),
            capability_proxy_samples=int(getattr(settings, "capability_proxy_samples", 20)),
            capability_proxy_fewshot=int(getattr(settings, "capability_proxy_fewshot", 5)),
            print_residual_geometry=bool(getattr(settings, "print_residual_geometry", False)),
            plot_residuals=bool(getattr(settings, "plot_residuals", False)),
            residual_plot_path=residual_plot_path,
            residual_plot_title=str(getattr(settings, "residual_plot_title", NativeConfig.residual_plot_title)),
            residual_plot_style=str(getattr(settings, "residual_plot_style", "dark_background")),
            scorers=scorers,
            scorer_settings=scorer_settings,
            max_response_length=int(getattr(settings, "abliteration_max_response_length", 64) or 64),
            max_weight_limit=max_weight_limit,
            direction_source=direction_source,
            evaluation_pruning=bool(getattr(settings, "abliteration_pruning", True)),
            search_seeds=bool(getattr(settings, "search_seeds", True)),
            print_debug_information=bool(getattr(settings, "print_debug_information", False)),
            reproducibility_information=raw_repro_info,
            ignore_mismatches=getattr(settings, "ignore_mismatches", None),
            export_strategy=export_strategy,
            max_shard_size=getattr(settings, "max_shard_size", "5GB"),
        )

    def update_from_dict(self, overrides: dict) -> None:
        """
        Applies overrides to this config, coercing serialized values (enums as
        strings, dataclasses as dicts, lists of dataclasses) back to their
        original types. None values are skipped so local paths survive.
        """
        for key, value in overrides.items():
            if not hasattr(self, key) or value is None:
                continue
            current = getattr(self, key)
            setattr(self, key, self._coerce_value(current, value))

    @staticmethod
    def _coerce_value(current: Any, value: Any) -> Any:
        if isinstance(current, Enum) and not isinstance(value, Enum):
            try:
                return type(current)(value)
            except ValueError:
                return value
        if is_dataclass(current) and not isinstance(current, type) and isinstance(value, dict):
            coerced = {}
            current_fields = {f.name: f for f in fields(current)}
            for field_key, field_value in value.items():
                if field_key not in current_fields:
                    continue
                default = getattr(current, field_key)
                coerced[field_key] = NativeConfig._coerce_value(default, field_value)
            return type(current)(**coerced)
        if isinstance(current, list) and isinstance(value, list):
            template = current[0] if current else None
            if (
                template is not None
                and is_dataclass(template)
                and not isinstance(template, type)
            ):
                return [
                    NativeConfig._coerce_value(template, item) if isinstance(item, dict) else item
                    for item in value
                ]
            return value
        if isinstance(current, tuple) and isinstance(value, list):
            return tuple(value)
        return value
