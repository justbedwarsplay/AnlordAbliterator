# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for export strategy, scorer settings plumbing, and config coercion."""

import json
from pathlib import Path

import pytest

from anlord.config import Settings
from anlord.native.config import (
    DatasetSpecification,
    ExportStrategy,
    NativeConfig,
    RowNormalization,
    ScorerConfig,
)


class TestExportStrategySettings:
    def test_default_is_merge(self):
        settings = Settings()
        assert settings.export_strategy == "merge"
        cfg = NativeConfig.from_anlord_settings(settings)
        assert cfg.export_strategy == ExportStrategy.MERGE

    def test_adapter_strategy(self):
        settings = Settings(export_strategy="adapter")
        cfg = NativeConfig.from_anlord_settings(settings)
        assert cfg.export_strategy == ExportStrategy.ADAPTER

    def test_invalid_strategy_rejected(self):
        with pytest.raises(ValueError, match="export_strategy"):
            Settings(export_strategy="bogus")

    def test_max_shard_size_passes_through(self):
        settings = Settings(max_shard_size="2GB")
        cfg = NativeConfig.from_anlord_settings(settings)
        assert cfg.max_shard_size == "2GB"

    def test_settings_round_trip_with_new_fields(self):
        settings = Settings(
            export_strategy="adapter",
            reproducibility_information="basic",
            print_residual_geometry=True,
            plot_residuals=True,
            scorers=[{"plugin": "anlord.scorers.kl_divergence.KLDivergence",
                      "optimization": "maximize"}],
            scorer_settings={"KLDivergence": {"score_name": "KL"}},
        )
        restored = Settings.from_dict(settings.to_dict())
        assert restored.export_strategy == "adapter"
        assert restored.reproducibility_information == "basic"
        assert restored.print_residual_geometry is True
        assert restored.scorers[0]["optimization"] == "maximize"
        cfg = NativeConfig.from_anlord_settings(restored)
        assert cfg.scorers[0].optimization == "maximize"
        assert cfg.scorer_settings["KLDivergence"]["score_name"] == "KL"


class TestScorerConfig:
    def test_validate_directions(self):
        for optimization in ("minimize", "maximize", "none"):
            ScorerConfig(optimization=optimization).validate()

    def test_validate_rejects_bad_direction(self):
        with pytest.raises(ValueError):
            ScorerConfig(optimization="sideways").validate()

    def test_validate_rejects_bad_instance_name(self):
        for bad in ("", "a b", "a.b"):
            with pytest.raises(ValueError):
                ScorerConfig(instance_name=bad).validate()


class TestResidualAnalysisSettings:
    def test_defaults(self):
        cfg = NativeConfig.from_anlord_settings(Settings())
        assert cfg.print_residual_geometry is False
        assert cfg.plot_residuals is False
        # Default plot location lives inside the output dir.
        assert cfg.residual_plot_path == str(Path("./output") / "plots")

    def test_explicit_plot_path_kept(self):
        cfg = NativeConfig.from_anlord_settings(
            Settings(residual_plot_path="C:/plots")
        )
        assert cfg.residual_plot_path == "C:/plots"


class TestExplicitCliOverrides:
    """Interactive mode must not silently ignore explicitly passed CLI args."""

    def _interactive_settings(self):
        """Settings as the interactive prompter would produce them."""
        return Settings(
            model="qwen/qwen3.5-0.8B",
            output_dir="C:/models/out",
            benchmarks=[],
            skip_benchmarks=True,
            mode="quick",
            limit=100,
            abliteration_trials=252,
            seed=42,
        )

    def _args(self, **kwargs):
        from anlord.cli.main import create_parser

        argv = ["--model", "ignored"]  # model forces non-interactive; irrelevant here
        for key, value in kwargs.items():
            if value is None:
                continue  # parser defaults already are None
            flag = key.replace("_", "-")
            if value is True:
                argv.append(f"--{flag}")
            elif value is False:
                argv.append(f"--no-{flag}")
            else:
                argv.extend([f"--{flag}", str(value)])
        parser = create_parser()
        # Remove the model so the namespace mimics an interactive invocation.
        namespace = parser.parse_args(argv)
        namespace.model = None
        namespace.batch_size = kwargs.get("batch_size", None)
        return namespace

    def test_batch_size_override_applied(self):
        from anlord.cli.main import apply_explicit_cli_overrides

        args = self._args(batch_size=128)
        settings = apply_explicit_cli_overrides(self._interactive_settings(), args)
        assert settings.abliteration_batch_size == 128

    def test_interactive_choices_survive_without_explicit_args(self):
        from anlord.cli.main import apply_explicit_cli_overrides

        args = self._args(batch_size=None)
        settings = apply_explicit_cli_overrides(self._interactive_settings(), args)
        # Chosen interactively: 252 trials, skipped benchmarks — untouched.
        assert settings.abliteration_trials == 252
        assert settings.skip_benchmarks is True
        assert settings.abliteration_batch_size is None  # stays auto

    def test_explicit_flags_override(self):
        from anlord.cli.main import apply_explicit_cli_overrides

        args = self._args(
            batch_size=64,
            export_strategy="adapter",
            reproducibility_info="none",
            ignore_mismatches=True,
            plot_residuals=True,
            skip_benchmarks=True,
        )
        settings = apply_explicit_cli_overrides(self._interactive_settings(), args)
        assert settings.abliteration_batch_size == 64
        assert settings.export_strategy == "adapter"
        assert settings.reproducibility_information == "none"
        assert settings.ignore_mismatches is True
        assert settings.plot_residuals is True
        assert settings.skip_benchmarks is True

    def test_resulting_config_uses_override(self):
        from anlord.cli.main import apply_explicit_cli_overrides
        from anlord.native.config import NativeConfig

        args = self._args(batch_size=128)
        settings = apply_explicit_cli_overrides(self._interactive_settings(), args)
        cfg = NativeConfig.from_anlord_settings(settings)
        assert cfg.batch_size == 128  # autotune skipped


class TestEvalPromptsWiring:
    """The classic --eval-prompts knob must reach the built-in scorers."""

    def test_default_injection(self):
        cfg = NativeConfig.from_anlord_settings(Settings())
        rate_table = cfg.scorer_settings["KeywordRate"]
        kl_table = cfg.scorer_settings["KLDivergence"]
        assert rate_table["prompts"]["split"] == "test[:100]"
        assert rate_table["prompts"]["dataset"] == "mlabonne/harmful_behaviors"
        assert kl_table["prompts"]["split"] == "test[:100]"
        assert kl_table["prompts"]["dataset"] == "mlabonne/harmless_alpaca"

    def test_eval_prompts_count_applies(self):
        settings = Settings(abliteration_evaluation_prompts=50)
        cfg = NativeConfig.from_anlord_settings(settings)
        assert cfg.scorer_settings["KeywordRate"]["prompts"]["split"] == "test[:50]"
        assert cfg.scorer_settings["KLDivergence"]["prompts"]["split"] == "test[:50]"

    def test_user_table_not_overridden(self):
        settings = Settings(
            abliteration_evaluation_prompts=50,
            scorer_settings={"KeywordRate": {"prompts": {"dataset": "x/y", "split": "train[:9]", "column": "text"}}},
        )
        cfg = NativeConfig.from_anlord_settings(settings)
        assert cfg.scorer_settings["KeywordRate"]["prompts"]["split"] == "train[:9]"
        # KLDivergence had no user table -> default injected.
        assert cfg.scorer_settings["KLDivergence"]["prompts"]["split"] == "test[:50]"

    def test_external_plugins_get_no_injection(self):
        settings = Settings(
            scorers=[{"plugin": "my_pkg.my_scorer.MyScorer", "optimization": "none"}],
        )
        cfg = NativeConfig.from_anlord_settings(settings)
        assert "KeywordRate" not in cfg.scorer_settings
        assert "KLDivergence" not in cfg.scorer_settings


class TestNativeConfigCoercion:
    """update_from_dict must restore serialized values with correct types."""

    def test_enum_coercion(self):
        cfg = NativeConfig()
        cfg.update_from_dict({"row_normalization": "pre", "quantization": "bnb_4bit",
                              "export_strategy": "adapter"})
        assert cfg.row_normalization == RowNormalization.PRE
        assert cfg.quantization.value == "bnb_4bit"
        assert cfg.export_strategy == ExportStrategy.ADAPTER

    def test_dataset_specification_coercion(self):
        cfg = NativeConfig()
        spec_dict = {"dataset": "some/dataset", "split": "train[:10]", "column": "text",
                     "commit": "abc123"}
        cfg.update_from_dict({"good_prompts": spec_dict})
        assert isinstance(cfg.good_prompts, DatasetSpecification)
        assert cfg.good_prompts.dataset == "some/dataset"
        assert cfg.good_prompts.commit == "abc123"
        # Unknown keys are dropped, defaults survive.
        assert cfg.good_prompts.prefix == ""

    def test_scorer_list_coercion(self):
        cfg = NativeConfig()
        cfg.update_from_dict({"scorers": [
            {"plugin": "anlord.scorers.keyword_rate.KeywordRate", "optimization": "maximize"},
        ]})
        assert isinstance(cfg.scorers[0], ScorerConfig)
        assert cfg.scorers[0].optimization == "maximize"

    def test_none_values_skipped(self):
        cfg = NativeConfig(model="original", batch_size=4)
        cfg.update_from_dict({"model": None, "batch_size": None})
        assert cfg.model == "original"
        assert cfg.batch_size == 4


class TestComponentIncludeList:
    """--components include list: parsing, mapping, and display plumbing."""

    def test_settings_string_normalization(self):
        settings = Settings(abliteration_components="attn.o_proj, mlp")
        assert settings.abliteration_components == ["attn.o_proj", "mlp"]
        with pytest.raises(ValueError, match="cannot be empty"):
            Settings(abliteration_components=[])
        with pytest.raises(ValueError, match="cannot be empty"):
            Settings(abliteration_components=" , ")

    def test_cli_parsing(self):
        from anlord.cli.main import create_parser, run_from_args

        parser = create_parser()
        args = parser.parse_args(["--model", "m", "--components", "attn"])
        settings = run_from_args(args)
        assert settings.abliteration_components == ["attn"]

        cfg = NativeConfig.from_anlord_settings(settings)
        assert cfg.abliteration_components == ["attn"]

    def test_no_flag_means_all_components(self):
        from anlord.cli.main import create_parser, run_from_args

        parser = create_parser()
        args = parser.parse_args(["--model", "m"])
        settings = run_from_args(args)
        assert settings.abliteration_components is None
        cfg = NativeConfig.from_anlord_settings(settings)
        assert cfg.abliteration_components is None

    def test_interactive_override(self):
        from anlord.cli.main import apply_explicit_cli_overrides

        interactive = Settings(model="m", abliteration_trials=50)
        parser = __import__("anlord.cli.main", fromlist=["create_parser"]).create_parser()
        args = parser.parse_args(["--model", "ignored", "--components", "attn.o_proj"])
        args.model = None
        settings = apply_explicit_cli_overrides(interactive, args)
        assert settings.abliteration_components == ["attn.o_proj"]

    def test_recorded_in_config_and_reproduction(self):
        """The filter must survive serialization everywhere reproduction needs it."""
        settings = Settings(abliteration_components=["attn"])
        cfg = NativeConfig.from_anlord_settings(settings)
        data = cfg.to_dict()
        assert data["abliteration_components"] == ["attn"]

        restored = NativeConfig()
        restored.update_from_dict(data)
        assert restored.abliteration_components == ["attn"]


class TestCliParsing:
    def test_new_flags_accepted(self, monkeypatch):
        from anlord.cli.main import create_parser, run_from_args

        parser = create_parser()
        args = parser.parse_args([
            "--model", "example/model",
            "--export-strategy", "adapter",
            "--print-residual-geometry",
            "--plot-residuals",
            "--max-shard-size", "2GB",
            "--reproducibility-info", "basic",
            "--reproduce", "reproduce.json",
            "--no-ignore-mismatches",
            "--print-debug-information",
        ])
        settings = run_from_args(args)
        assert settings.export_strategy == "adapter"
        assert settings.print_residual_geometry is True
        assert settings.plot_residuals is True
        assert settings.max_shard_size == "2GB"
        assert settings.reproducibility_information == "basic"
        assert settings.reproduce == "reproduce.json"
        assert settings.ignore_mismatches is False
        assert settings.print_debug_information is True

    def test_scorers_inline_json(self):
        from anlord.cli.main import create_parser, run_from_args

        parser = create_parser()
        scorers = [{"plugin": "anlord.scorers.kl_divergence.KLDivergence",
                    "optimization": "none"}]
        args = parser.parse_args(["--model", "m", "--scorers", json.dumps(scorers)])
        settings = run_from_args(args)
        assert settings.scorers == scorers

    def test_scorers_json_file(self, tmp_path):
        from anlord.cli.main import create_parser, run_from_args

        scorers = [{"plugin": "anlord.scorers.keyword_rate.KeywordRate",
                    "optimization": "minimize"}]
        path = tmp_path / "scorers.json"
        path.write_text(json.dumps(scorers), encoding="utf-8")
        parser = create_parser()
        args = parser.parse_args(["--model", "m", "--scorers", str(path)])
        settings = run_from_args(args)
        assert settings.scorers == scorers

    def test_scorers_invalid_rejected(self):
        from anlord.cli.main import create_parser, run_from_args

        parser = create_parser()
        args = parser.parse_args(["--model", "m", "--scorers", "not-json"])
        with pytest.raises(ValueError, match="Invalid --scorers"):
            run_from_args(args)
