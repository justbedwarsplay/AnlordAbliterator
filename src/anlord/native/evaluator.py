# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Native evaluator — manages evaluation of the model using configured scorer plugins.

Loads scorers, establishes baseline scores, and runs scorers during optimization.
Each scorer is an independent objective candidate: the Optuna study directions
are derived from the configured scorers via `get_objective_names()` /
`get_objective_directions()`, and objective values per trial come from
`get_objective_values()`.

For backwards compatibility with the fixed refusals/KL pipeline, the evaluator
also extracts legacy metrics (integer refusal counts, raw KL value) from the
scorer scores whenever matching built-in scorers are loaded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from optuna.study import StudyDirection

from .config import DatasetSpecification, NativeConfig, ScorerConfig
from .model import Model
from .plugin import Context, get_plugin_namespace, is_builtin_plugin, load_plugin
from .prompts import Prompt
from .scorer import Score, Scorer
from .utils import deep_merge_dicts, parse_study_direction, print


@dataclass
class ScorerEntry:
    scorer: Scorer
    name: str
    config: ScorerConfig


class Evaluator:
    settings: NativeConfig
    model: Model

    # Legacy metrics (integer refusals / raw KL) extracted from the scorer
    # scores, provided for compatibility with the pre-scorer pipeline
    # (they back the NativeAbliteratorResult fields).
    base_refusals: int = 0
    base_total_prompts: int = 0
    base_kl_divergence: float = 0.0
    last_refusals: Optional[int] = None
    last_total_prompts: Optional[int] = None
    last_kl_divergence: Optional[float] = None

    def __init__(self, settings: NativeConfig, model: Model):
        self.settings = settings
        self.model = model
        self._scorer_entries: list[ScorerEntry] = []

        print()
        print("Loading and initializing scorers...")
        self._load_and_init_scorers()

        print()
        print("Getting baseline scores...")
        self.baseline_scores = self.get_baseline_scores()
        for name, score in self.baseline_scores:
            print(f"* Baseline [bold]{name}:[/] [green]{score.rich_display}[/]")

        self._extract_legacy_metrics(self.baseline_scores, baseline=True)

    # ------------------------------------------------------------------
    # Scorer loading
    # ------------------------------------------------------------------

    def _load_and_init_scorers(self) -> None:
        """
        Load and instantiate all configured scorer plugins,
        then run their initialization hooks.
        """
        scorer_configs = self.settings.scorers
        if not scorer_configs:
            raise ValueError("No scorers configured. Set 'scorers' in the config")

        scorer_keys: set[str] = set()

        # Resolve plugin classes from names and validate.
        for raw_config in scorer_configs:
            # Accept dicts (as they appear in serialized configs) and plain
            # plugin names in addition to ScorerConfig instances.
            if isinstance(raw_config, str):
                raw_config = {"plugin": raw_config}
            if isinstance(raw_config, dict):
                raw_config = ScorerConfig(**raw_config)
            config: ScorerConfig = raw_config
            config.validate()
            scorer_cls = load_plugin(name=config.plugin, base_class=Scorer)
            scorer_cls.validate_contract()

            print(
                f"* Loaded: [bold]{scorer_cls.__name__} "
                f"{'- ' + config.instance_name if config.instance_name else ''}[/bold]"
            )

            # Instantiate scorers.
            instance_name = config.instance_name or None

            raw_settings = self._get_scorer_settings_raw(
                scorer_cls=scorer_cls, instance_name=instance_name
            )
            scorer_settings = scorer_cls.validate_settings(raw_settings)

            scorer = scorer_cls(
                anlord_settings=self.settings,
                settings=scorer_settings,
            )

            # External labeling key: ensures multiple instances can coexist.
            scorer_key = (
                scorer_cls.__name__
                if not instance_name
                else f"{scorer_cls.__name__}_{instance_name}"
            )
            if scorer_key in scorer_keys:
                raise ValueError(
                    f"Duplicate scorer instance name: {scorer_key}. "
                    "Give each instance a unique `instance_name`."
                )
            scorer_keys.add(scorer_key)

            scorer_instance_name = (
                f"{scorer.score_name} - {instance_name}"
                if instance_name
                else scorer.score_name
            )
            self._scorer_entries.append(
                ScorerEntry(scorer=scorer, config=config, name=scorer_instance_name)
            )

        # Run scorer init hooks.
        ctx = Context(settings=self.settings, model=self.model)

        for entry in self._scorer_entries:
            entry.scorer.init(ctx)

    def _get_scorer_settings_raw(
        self, *, scorer_cls: type[Scorer], instance_name: str | None
    ) -> dict[str, Any]:
        """
        Build the raw settings dict for a scorer class and optional instance.

        Config rules:
        - Base settings live in `scorer_settings["<ClassName>"]`
          (applies to all instances).
        - Instance overrides live in `scorer_settings["<ClassName>_<instance>"]`
          (preferred).
        - Only merge/validate keys that exist in the scorer Settings schema.
        """
        settings_model = scorer_cls.get_settings_model()
        if settings_model is None:
            # No settings schema: nothing to merge/validate.
            return {}

        class_name = scorer_cls.__name__

        namespaces = [f"{class_name}"]
        if instance_name:
            namespaces.append(f"{class_name}_{instance_name}")

        merged_settings: dict[str, Any] = {}
        allowed_keys = set(settings_model.model_fields.keys())

        for namespace in namespaces:
            raw_table = get_plugin_namespace(self.settings.scorer_settings, namespace)
            filtered = {k: v for k, v in raw_table.items() if k in allowed_keys}
            merged_settings = deep_merge_dicts(merged_settings, filtered)

        return merged_settings

    # ------------------------------------------------------------------
    # Dataset specifications / reproducibility
    # ------------------------------------------------------------------

    def get_dataset_specifications(self) -> list[DatasetSpecification]:
        """
        Collect the dataset specifications declared in the settings of all
        loaded scorers.
        """
        specifications = []
        for entry in self._scorer_entries:
            if entry.scorer.settings is None:
                continue
            for value in dict(entry.scorer.settings).values():
                if isinstance(value, DatasetSpecification):
                    specifications.append(value)
        return specifications

    def all_scorers_reproducible(self) -> bool:
        """
        Returns True if all scorers are reproducible, False if not.
        """
        return all(entry.scorer.reproducible for entry in self._scorer_entries)

    def all_scorers_builtin(self) -> bool:
        """
        Returns True if all scorers are built-in, i.e. shipped with
        Anlord Abliterator by default.
        """
        return all(
            is_builtin_plugin(entry.config.plugin) for entry in self._scorer_entries
        )

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def get_scores(self) -> list[tuple[str, Score]]:
        """
        Run all scorers and return their scores and names.

        Returns:
            List of `Score` from each scorer and its name.
        """
        ctx = Context(settings=self.settings, model=self.model)
        scores = [
            (entry.name, entry.scorer.get_score(ctx)) for entry in self._scorer_entries
        ]
        self._extract_legacy_metrics(scores, baseline=False)
        return scores

    def get_baseline_scores(self) -> list[tuple[str, Score]]:
        """
        Run all scorers and return their baseline scores and names.

        Returns:
            List of `Score` from each scorer and its name.
        """
        ctx = Context(settings=self.settings, model=self.model)
        return [
            (entry.name, entry.scorer.get_baseline_score(ctx))
            for entry in self._scorer_entries
        ]

    def get_paired_score_records(
        self, scores: list[tuple[str, Score]]
    ) -> list[dict[str, Any]]:
        """
        Pair each trial score with its baseline into one serializable record.

        `scores` (from `get_scores()`) and `self.baseline_scores` are both ordered
        by `_scorer_entries`, so they align positionally.
        """
        records: list[dict[str, Any]] = []
        for (name, score), (baseline_name, baseline) in zip(
            scores, self.baseline_scores
        ):
            assert name == baseline_name, (
                f"Score/baseline order mismatch: {name!r} != {baseline_name!r}"
            )
            records.append(
                {
                    "name": name,
                    "score": dict(score.__dict__),
                    "baseline": dict(baseline.__dict__),
                }
            )
        return records

    # ------------------------------------------------------------------
    # Objectives
    # ------------------------------------------------------------------

    def _objective_entries(self) -> list[ScorerEntry]:
        """
        Scorer entries that participate in optimization, in canonical order.
        Single source of truth for which scorers are objectives and in what
        order. Every objective-derived list (names, directions, values) is built
        from this so they stay positionally aligned: Optuna matches the objective
        values returned each trial to the study `directions` by index, so a length
        or order mismatch here would silently corrupt the optimization.
        """
        return [
            entry
            for entry in self._scorer_entries
            if parse_study_direction(entry.config.optimization) != StudyDirection.NOT_SET
        ]

    def get_objective_names(self) -> list[str]:
        """Return objective names for scores used in optimization."""
        return [entry.name for entry in self._objective_entries()]

    def get_objective_values(
        self, scores: list[tuple[str, Score]]
    ) -> tuple[float, ...]:
        """
        Extract objective values as a tuple for Optuna.

        Ordered by `_objective_entries()` so the result aligns by index with
        `get_objective_names()` and `get_objective_directions()`.
        """
        score_by_name = {name: score for name, score in scores}
        return tuple(
            score_by_name[entry.name].value for entry in self._objective_entries()
        )

    def get_objective_directions(self) -> list[StudyDirection]:
        """Get optimization directions for objectives."""
        return [
            parse_study_direction(entry.config.optimization)
            for entry in self._objective_entries()
        ]

    # ------------------------------------------------------------------
    # Legacy metrics bridge (integer refusals / raw KL for the pipeline)
    # ------------------------------------------------------------------

    def _extract_legacy_metrics(
        self, scores: list[tuple[str, Score]], baseline: bool
    ) -> None:
        """
        Extract integer refusal counts and the raw KL divergence from the most
        recent scores, so the rest of the pipeline (result files, comparisons)
        keeps working with the plain refusals/KL metrics.
        """
        scorer_by_name = {entry.name: entry.scorer for entry in self._scorer_entries}
        for name, _score in scores:
            scorer = scorer_by_name.get(name)
            if scorer is None:
                continue
            if baseline:
                if hasattr(scorer, "last_match_count"):
                    self.base_refusals = int(scorer.last_match_count)
                    self.base_total_prompts = int(scorer.last_total)
                if hasattr(scorer, "last_kl_value"):
                    self.base_kl_divergence = float(scorer.last_kl_value)
            else:
                if hasattr(scorer, "last_match_count"):
                    self.last_refusals = int(scorer.last_match_count)
                    self.last_total_prompts = int(scorer.last_total)
                if hasattr(scorer, "last_kl_value"):
                    self.last_kl_divergence = float(scorer.last_kl_value)

    @property
    def bad_prompts(self) -> list[Prompt]:
        """Prompts used for refusal counting (from the first keyword-rate scorer)."""
        for entry in self._scorer_entries:
            prompts = getattr(entry.scorer, "prompts", None)
            if prompts and hasattr(entry.scorer, "last_match_count"):
                return prompts
        return []
